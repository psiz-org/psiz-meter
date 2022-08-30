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
"""Meter module.

Functions:
    ensemble_marginalized_similarity

"""

import numpy as np
import psiz
import tensorflow as tf

from psiz_meter.meters.rho_metric import rho_metric


def ensemble_marginalized_similarity(
        fp_psiz_model_list, ds_pairs, compute_within=False,
        mode='pearson', n_posterior_sample=300, verbose=0):
    """Ensemble-based similarity matrix.

    Loads and evaluates models in list one at a time in order to
    conserve memory resources.

    Arguments:
        fp_psiz_model_list:
        ds_pairs:
        compute_within (optional):
        mode (optional):
        n_posterior_sample (optional):
        verbose (optional):

    """
    n_model = len(fp_psiz_model_list)

    smat_list = []
    for fp_model in fp_psiz_model_list:
        model = tf.keras.models.load_model(fp_model)
        sim_arr = psiz.utils.pairwise_similarity(
            model.stimuli, model.kernel, ds_pairs,
            n_sample=n_posterior_sample, compute_average=True,
            verbose=verbose
        )
        smat_list.append(sim_arr)

    # Optionally, compute average correlations between models.
    if compute_within:
        rho_within = []
        idx_i, idx_j = np.triu_indices(n_model, k=1)
        for idx_a, idx_b in zip(idx_i, idx_j):
            rho_within.append(
                rho_metric(smat_list[idx_a], smat_list[idx_b], mode=mode)
            )
        rho_within = np.mean(rho_within)
    else:
        rho_within = None

    # Compute ensemble average.
    # NOTE: Since the number of samples is the same for each model, the
    # average of average is equivalent to taking the average of all samples.
    smat = tf.stack(smat_list, axis=0)
    smat = tf.reduce_mean(smat, axis=0)

    return smat, rho_within
