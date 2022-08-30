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
"""Meters module.

Functions:
    basic_similarity

"""

import logging
import time

import numpy as np
import psiz
import tensorflow as tf


def basic_similarity(z, ds_pairs, similarity='cosine', verbose=0):
    """Compute pairwise similarity for a dataset of pairs.

    Arguments:
        z: Embedding point-estimate.
        ds_pairs: A dataset of index pairs.
        similarity (optional): A string indicating a similarity
            function.
        verbose (optional): Non-negative integer indicating verbosity
            of output.

    Returns:
        s_arr: Corresponding similarity for incoming pairs.

    """
    start_s = time.time()
    s_arr = []

    if verbose > 0:
        n_batch = 0
        for _ in ds_pairs:
            n_batch += 1
        progbar = psiz.utils.ProgressBarRe(
            n_batch, prefix='Similarity:', length=50
        )
        progbar.update(0)
        progbar_counter = 0
        # Determine how often progbar should update (we use 50 since that is
        # the visual length of the progbar).
        progbar_update = np.maximum(1, int(np.ceil(n_batch / 50)))

    for x_batch in ds_pairs:
        z_0 = tf.gather(z, x_batch[0])
        z_1 = tf.gather(z, x_batch[1])
        if similarity == 'cosine':
            s_arr.append(
                tf.negative(
                    tf.keras.losses.cosine_similarity(
                        z_0, z_1, axis=1
                    )
                )
            )
        elif similarity == 'exp':
            s_arr.append(
                tf.exp(-tf.sqrt(tf.reduce_sum(tf.pow(z_0 - z_1, 2), axis=1)))
            )
        elif similarity == 'dot':
            s_arr.append(
                tf.reduce_sum(z_0 * z_1, axis=1)
            )
        else:
            raise NotImplementedError('Requested similarity not implemented.')

        if verbose > 0:
            if (np.mod(progbar_counter, progbar_update) == 0):
                progbar.update(progbar_counter + 1)
            progbar_counter += 1

    duration_s = time.time() - start_s
    if verbose > 0:
        progbar.update(n_batch)
        logging.info('Finished evaluation {0:.2f} s'.format(duration_s))
    start_s = time.time()
    # NOTE: tf.concat is very fast.
    s_arr = tf.concat(s_arr, 0).numpy()
    duration_s = time.time() - start_s

    return s_arr
