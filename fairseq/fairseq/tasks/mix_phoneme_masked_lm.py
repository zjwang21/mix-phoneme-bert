# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from omegaconf import II, MISSING, OmegaConf

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MixPhonemeMaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    SortDataset,
    MixPhonemeTokenBlockDataset,
    data_utils,
    FlattenDataset,
    StripTokenDataset
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES

logger = logging.getLogger(__name__)


@dataclass
class MixPhonemeMaskedLMConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")

    include_target_tokens: bool = field(
        default=False,
        metadata={
            "help": "include target tokens in model input. this is used for data2vec"
        },
    )
    include_index: bool = field(
        default=True,
        metadata={"help": "include index in model input. this is used for data2vec"},
    )
    skip_masking: bool = field(
        default=False,
        metadata={"help": "skip masking at dataset"},
    )
    # subsample_train: float = field(
    #     default=1,
    #     metadata={"help": "shorten training set for debugging"},
    # )
    d2v2_multi: bool = field(
        default=False,
        metadata={"help": "prepare dataset for data2vec_multi"},
    )


@register_task("mix_phoneme_masked_lm", dataclass=MixPhonemeMaskedLMConfig)
class MixPhonemeMaskedLMTask(FairseqTask):

    cfg: MixPhonemeMaskedLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: MixPhonemeMaskedLMConfig, dictionary_p=None, dictionary_sp=None, dictionary_mask=None):
        super().__init__(cfg)
        #self.dictionary = dictionary or self.load_dict(cfg)
        self.dictionary_p = dictionary_p
        self.dictionary_sp = dictionary_sp
        self.dictionary_mask = dictionary_mask
        # add mask token
        self.mask_idx_p = self.dictionary_p.add_symbol("<mask>")
        self.mask_idx_sp = self.dictionary_sp.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, cfg: MixPhonemeMaskedLMConfig, **kwargs):
        dictionary_p, dictionary_sp, dictionary_mask = cls.load_dict(cfg)
        return cls(cfg, dictionary_p=dictionary_p, dictionary_sp=dictionary_sp, dictionary_mask=dictionary_mask)

    @classmethod
    def load_dict(cls, cfg):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary_p = Dictionary.load(os.path.join(paths[0], "dict.p.txt"))
        dictionary_sp = Dictionary.load(os.path.join(paths[0], "dict.sp.txt"))
        dictionary_mask = Dictionary.load(os.path.join(paths[0], "dict.wwm.txt"))
        logger.info("phoneme dictionary: {} types".format(len(dictionary_p)))
        logger.info("sup-phoneme dictionary: {} types".format(len(dictionary_sp)))
        return dictionary_p, dictionary_sp, dictionary_mask

    def _load_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path_sp = os.path.join(data_path, split + '.sp-sp.sp')
        split_path_p = os.path.join(data_path, split + '.p-p.p')   #因素和sup因素数据导入
        dataset_impl = 'mmap'
        split_path_wwm = os.path.join(data_path, split + '.wwm-wwm.wwm')   #因素和sup因素数据导入
        dataset_sp = data_utils.load_indexed_dataset(
            split_path_sp,
            self.source_dictionary[1],
            dataset_impl=dataset_impl,          
            combine=combine,
        )
        if dataset_sp is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path_sp)
            )
        dataset_p = data_utils.load_indexed_dataset(
            split_path_p,
            self.source_dictionary[0],
            dataset_impl=dataset_impl,          
            combine=combine,
        )
        if dataset_p is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path_p)
            )
        assert len(dataset_p) == len(dataset_sp)

        dataset_wwm = data_utils.load_indexed_dataset(
            split_path_wwm,
            self.mask_dictionary,
            dataset_impl=dataset_impl,
            combine=combine,
        )
        if dataset_wwm is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path_wwm)
            )
        assert len(dataset_sp) == len(dataset_wwm)

        dictionary = self.source_dictionary
        assert dictionary[0].pad() == dictionary[1].pad()
        assert dictionary[0].eos() == dictionary[1].eos()
        assert dictionary[0].bos() == dictionary[1].bos()

        dataset_sp = StripTokenDataset(dataset_sp, dictionary[0].eos())
        dataset_p = StripTokenDataset(dataset_p, dictionary[0].eos())
        dataset_wwm = StripTokenDataset(dataset_wwm, dictionary[0].eos())

        # create continuous blocks of tokens
        dataset = MixPhonemeTokenBlockDataset(
            dataset_sp,
            dataset_p,
            dataset_wwm,
            dataset_sp.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary[0].pad(),
            eos=self.source_dictionary[0].eos(),
            bos=self.source_dictionary[0].bos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {} and {} respectively".format(len(dataset), split_path_p, split_path_sp))

        return dataset
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        # return PrependTokenDataset(dataset, self.source_dictionary.bos())

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset = self._load_dataset_split(split, epoch, combine)

        # create masked input and targets
        #mask_whole_words = (
        #    get_whole_word_mask(self.cfg, self.source_dictionary)
        #    if self.cfg.mask_whole_words
        #    else None
        #)
        dictionary = self.source_dictionary[0]
        '''
        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return tok.startswith("\u2581")
            except ValueError:
                return True

        mask_whole_words = (
            torch.ByteTensor(list(map(is_beginning_of_word, range(len(dictionary)))))
            if self.cfg.mask_whole_words
            else None
        )'''

        mix_lm_dataset = MixPhonemeMaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary[0].pad(),
            mask_idx_p=self.mask_idx_p,
            mask_idx_sp=self.mask_idx_sp,
            seed=self.cfg.seed,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            mask_whole_words=True,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
            skip_masking=self.cfg.skip_masking,
        )

        sp_src_dataset = FlattenDataset(mix_lm_dataset, data_type='sp_src')
        p_src_dataset = FlattenDataset(mix_lm_dataset, data_type='p_src')
        sp_tgt_dataset = FlattenDataset(mix_lm_dataset, data_type='sp_tgt')
        p_tgt_dataset = FlattenDataset(mix_lm_dataset, data_type='p_tgt')
        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(mix_lm_dataset.dataset))

        #if self.cfg.d2v2_multi:
        #    dataset = self._d2v2_multi_dataset(src_dataset)
        #else:
        #    dataset = self._regular_dataset(src_dataset, target_dataset)
        dataset = self._regular_dataset(sp_src_dataset, sp_tgt_dataset, p_src_dataset, p_tgt_dataset)
        self.datasets[split] = SortDataset(
            dataset, sort_order=[shuffle, sp_src_dataset.sizes]
        )

    def _regular_dataset(self, sp_src_dataset, sp_tgt_dataset, p_src_dataset, p_tgt_dataset):
        sp_target_dataset = RightPadDataset(
            sp_tgt_dataset,
            pad_idx=self.source_dictionary[0].pad(),
        )
        p_target_dataset = RightPadDataset(
            p_tgt_dataset,
            pad_idx=self.source_dictionary[0].pad(),
        )
        input_dict = {
            "src_tokens":{
                "sp_src_tokens": RightPadDataset(
                    sp_src_dataset,
                    pad_idx=self.source_dictionary[0].pad(),
                ),
                "p_src_tokens": RightPadDataset(
                    p_src_dataset,
                    pad_idx=self.source_dictionary[0].pad(),
                ),
            },
            "src_lengths": NumelDataset(sp_src_dataset, reduce=False),
        }
        #if self.cfg.include_target_tokens:
        #    input_dict["target_tokens"] = target_dataset
        if self.cfg.include_index:
            input_dict["src_id"] = IdDataset()

        target_dict = {
            "sp_tgt_tokens": sp_target_dataset,
            "p_tgt_tokens": p_target_dataset,
        }

        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "target": target_dict,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(sp_src_dataset, reduce=True),
            },
            sizes=[sp_src_dataset.sizes],
        )
        return dataset

    def _d2v2_multi_dataset(self, src_dataset):
        input_dict = {
            "source": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "id": IdDataset(),
            "padding_mask": RightPaddingMaskDataset(src_dataset),
        }

        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        )
        return dataset
    # to do 

    def build_dataset_for_embedding(self, split, sort=False, batch=False):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[0]
        split_path_sp = os.path.join(data_path, split + '.sp-sp.sp')
        split_path_p = os.path.join(data_path, split + '.p-p.p')   #因素和sup因素数据导入
        dataset_sp = data_utils.load_indexed_dataset(
            split_path_sp,
            self.source_dictionary[1],
        )
        if dataset_sp is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path_sp)
            )
        dataset_p = data_utils.load_indexed_dataset(
            split_path_p,
            self.source_dictionary[0],
        )
        if dataset_p is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path_p)
            )
        assert len(dataset_p) == len(dataset_sp)
        dictionary = self.source_dictionary
        assert dictionary[0].pad() == dictionary[1].pad()
        assert dictionary[0].eos() == dictionary[1].eos()
        assert dictionary[0].bos() == dictionary[1].bos()
        # create continuous blocks of tokens
        logger.info("loaded {} samples from: {} and {} respectively".format(len(dataset_p), split_path_p, split_path_sp))
        dataset_p = PrependTokenDataset(dataset_p, self.source_dictionary[0].bos())
        dataset_sp = PrependTokenDataset(dataset_sp, self.source_dictionary[1].bos())
        if not batch:
            return dataset_p, dataset_sp
        input_dict = {
            "src_tokens":{
                "sp_src_tokens": RightPadDataset(
                    dataset_sp,
                    pad_idx=self.source_dictionary[0].pad(),
                ),
                "p_src_tokens": RightPadDataset(
                    dataset_p,
                    pad_idx=self.source_dictionary[0].pad(),
                ),
            },
            "src_lengths": NumelDataset(dataset_sp, reduce=False),
        }
        #if self.cfg.include_target_tokens:
        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(dataset_sp, reduce=True),
            },
            sizes=[dataset_sp.sizes],
        )
        return dataset
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        # return PrependTokenDataset(dataset, self.source_dictionary.bos())

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            MixPhonemeTokenBlockDataset(
                src_tokens,
                src_lengths,
                self.cfg.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return (self.dictionary_p, self.dictionary_sp)

    @property
    def target_dictionary(self):
        return (self.dictionary_p, self.dictionary_sp)

    def mask_dictionary(self):
        return self.dictionary_mask

    def begin_epoch(self, epoch, model):
        model.set_epoch(epoch)

    def max_positions(self):
        return self.cfg.tokens_per_sample
