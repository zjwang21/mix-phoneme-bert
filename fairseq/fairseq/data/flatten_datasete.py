# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class FlattenDataset(BaseWrapperDataset):
    def __init__(self, dataset, data_type=None):
        super().__init__(dataset)
        self.data_type = data_type

    def __getitem__(self, idx):
        return self.dataset[idx][self.data_type]
