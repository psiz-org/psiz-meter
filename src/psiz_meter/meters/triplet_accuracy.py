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
"""Meters Module.

Functions:
    triplet_accuracy

"""

import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances


def triplet_accuracy(z, obs, distance='l2', batch_size=10000):
    """Compute triplet accuracy.
    
    Args:
        z: The embedding to evaluate.
        obs: The triplet observations.
        distance (optional): The distance to use.
        batch_size (optional): The batch size.

    Returns:
        The triplet accuracy of the embedding.

    """
    n_batch = int(np.ceil(obs.n_trial / batch_size))

    d_func = None
    if distance == 'l1':
        def d_func(x, y):
            return np.sum(np.abs(x - y), axis=1)
    elif distance == 'l2':
        def d_func(x, y):
            return np.sqrt(np.sum((x - y)**2, axis=1))
    elif distance == 'cos':
        def d_func(x, y):
            return paired_cosine_distances(x, y)

    is_correct = []
    for i_batch in range(n_batch):
        batch_idx_start = i_batch*batch_size
        batch_idx_end = batch_idx_start + batch_size
        z_q = z[obs.stimulus_set[batch_idx_start:batch_idx_end, 0], :]
        z_1 = z[obs.stimulus_set[batch_idx_start:batch_idx_end, 1], :]
        z_2 = z[obs.stimulus_set[batch_idx_start:batch_idx_end, 2], :]        
        d_q1 = d_func(z_q, z_1)
        d_q2 = d_func(z_q, z_2)
        # NOTE: Since we are using distance, we check if d(q,1) < d(q,2).
        is_correct.append(np.less(d_q1, d_q2))

    is_correct = np.concatenate(is_correct, axis=0)
    return np.mean(is_correct)
