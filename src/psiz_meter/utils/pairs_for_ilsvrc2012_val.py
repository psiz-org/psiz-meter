# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Meter Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""

Functions:
    pairs_for_ilsvrc2012_val

"""

import numpy as np
from psiz.utils import pairwise_index_dataset


def pairs_for_ilsvrc2012_val(version_round):
    """Create TF dataset representing pairs of indices.

    The generated dataset depend on the version on ImageNet-HSJ
    requested.

    For versions >v0.1, not all pairs are used since evaluating all
    pairs is prohibitively expensive. Instead a random subset of pairs
    is selected. Mini-experiments indicated that using 10% of all
    possible pairs is a sufficient sample.

    Args:
        version_round: Integer indicating the round.

    Returns:
        A TF dataset of index pairs.

    """
    if version_round == 118:
        # Seed set (v0.1).
        batch_size = 10000
        n_stimuli = 1000
        # Create all pairs.
        # NOTE: We start at index "1" since, index "0" is a placeholder.
        ds_pairs, ds_info = pairwise_index_dataset(
            np.arange(1, n_stimuli + 1), batch_size=batch_size
        )
    elif version_round == 195:
        # Full validation set (v0.2).
        batch_size = 10000
        n_stimuli = 50000
        # Create pairs.
        # NOTE: sumbsample=.01 ~ 10s per target matrix, .1 ~150s per target
        # matrix.
        # NOTE: We start at index "1" since, index "0" is a placeholder.
        subsample = .1  # .03, .1
        ds_pairs, ds_info = pairwise_index_dataset(
            np.arange(1, n_stimuli + 1), batch_size=batch_size, subsample=subsample, seed=252
        )
    else:
        raise NotImplementedError('Requested round not implemented.')

    return ds_pairs
