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
    rank_to_triplets

"""

import numpy as np
import psiz


def rank_to_triplets(obs):
    """Convert rank observations to implied triplets."""
    # Check if all observations are 8-rank-2.
    if np.sum(np.not_equal(obs.n_reference, 8)) > 0:
        raise NotImplementedError(
            'Not all observations have 8 references.'
        )
    if np.sum(np.not_equal(obs.n_select, 2)) > 0:
        raise NotImplementedError(
            'Not all observations have 2 selections.'
        )

    mask_zero = obs.mask_zero

    # Pre-define all implied triplet cases for 8-rank-2 trials.
    # NOTE: The order reflect query, chosen reference, and non-chosen
    # reference.
    case_list = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 5],
        [0, 1, 6],
        [0, 1, 7],
        [0, 1, 8],
        [0, 2, 3],
        [0, 2, 4],
        [0, 2, 5],
        [0, 2, 6],
        [0, 2, 7],
        [0, 2, 8],
    ]
    obs_triplet_list = []
    for i_case in case_list:
        stimulus_set_triplets = obs.stimulus_set[:, i_case]

        # Perform some sanity checks.
        np.testing.assert_array_equal(
            stimulus_set_triplets[:, 0], obs.stimulus_set[:, i_case[0]]
        )
        np.testing.assert_array_equal(
            stimulus_set_triplets[:, 1], obs.stimulus_set[:, i_case[1]]
        )
        np.testing.assert_array_equal(
            stimulus_set_triplets[:, 2], obs.stimulus_set[:, i_case[2]]
        )

        n_select = np.ones([obs.n_trial], dtype=int)
        weight = obs.weight
        obs_triplet = psiz.trials.RankObservations(
            stimulus_set_triplets, n_select=None, weight=None,
            mask_zero=mask_zero
        )
        obs_triplet_list.append(obs_triplet)

    return psiz.trials.stack(obs_triplet_list)
