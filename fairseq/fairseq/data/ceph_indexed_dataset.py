# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from linecache import updatecache
import shutil
import struct
from functools import lru_cache
import boto3
import numpy as np
import torch
from fairseq.dataclass.constants import DATASET_IMPL_CHOICES
from fairseq.data.fasta_dataset import FastaDataset
from fairseq.file_io import PathManager
from fairseq.data.huffman import HuffmanMMapIndexedDataset, HuffmanMMapIndex

from . import FairseqDataset

from petrel_client.client import Client
client = Client('/mnt/lustre/wangzhijun2/ceph_conf/petreloss.conf')

from typing import Union
import logging
logger = logging.getLogger(__name__)
_code_to_dtype = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.double,
    8: np.uint16,
    9: np.uint32,
    10: np.uint64,
}

def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None):
    return CMMapIndexedDataset(path)


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"

def _warmup_mmap_file(stream):
    while stream.read(100 * 1024 * 1024):
        pass

class Ceph_client:
    def __init__(self, filepath) -> None:
        awsd = {
            "aws_access_key_id" : 'APC1S9X31G1ZGA3RPPQ0',
            "aws_secret_access_key" : '8xtDJHpHIowoLcDhubEcVt04eLURvvCtVJqCZ7p8',
            "endpoint_url" : 'http://10.5.41.190:80',
            "service_name" : 's3'
            }
        self.key = filepath
        self.client = boto3.client(**awsd)
        logger.info('create ceph client for {}'.format(filepath))
    def get(self):
        tmp = self.key.split("/")
        bucket = tmp[0]
        key = "/".join(tmp[1:])
        b = self.client.get_object(Bucket=bucket, Key=key, updatecache=True)
        return b["Body"]

class CMMapIndexedDataset(torch.utils.data.Dataset):
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"
        def __init__(self, idxkey):
            #此处流式读取，只用作校验，要关cache，不然报错
            stream = client.get(idxkey, no_cache=True, enable_stream=True)
            logger.info('reading ceph data {}'.format(idxkey))
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, (
                "Index file doesn't match expected format. "
                "Make sure that --dataset-impl is configured properly."
            )
            version = struct.unpack("<Q", stream.read(8))
            assert (1,) == version

            (dtype_code,) = struct.unpack("<B", stream.read(1))
            self._dtype = _code_to_dtype[dtype_code]
            self._dtype_size = self._dtype().itemsize

            self._len = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()
            
            self._bin_buffer_mmap = client.get(idxkey, update_cache=True)
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        idxkey = index_file_path(self._path)
        idxkey = idxkey.split('/')[-1]
        idxkey = "zk:s3://wzj/mixpb-bin-5kw/" + idxkey

        datakey = data_file_path(self._path)
        datakey = datakey.split('/')[-1]
        datakey = "zk:s3://wzj/mixpb-bin-5kw/" + datakey

        self._path = path
        self._index = self.Index(idxkey)

        logger.info('reading ceph data {}'.format(datakey))
        self._bin_buffer_mmap = client.get(datakey, update_cache=True)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return PathManager.exists(index_file_path(path)) and PathManager.exists(
            data_file_path(path)
        )

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        # TODO: a quick fix. make it a child class of FairseqDataset instead?
        return True

def get_indexed_dataset_to_local(path) -> str:
    local_index_path = PathManager.get_local_path(index_file_path(path))
    local_data_path = PathManager.get_local_path(data_file_path(path))

    assert local_index_path.endswith(".idx") and local_data_path.endswith(".bin"), (
        "PathManager.get_local_path does not return files with expected patterns: "
        f"{local_index_path} and {local_data_path}"
    )

    local_path = local_data_path[:-4]  # stripping surfix ".bin"
    assert local_path == local_index_path[:-4]  # stripping surfix ".idx"
    return local_path
