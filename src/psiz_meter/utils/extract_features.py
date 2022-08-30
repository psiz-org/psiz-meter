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
   extract_features

"""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from psiz_meter.utils.center_crop import center_crop


def extract_features(fp_list, model, n_stimuli, n_dim, img_size, preprocess_input, batch_size=32):
    model_features = []
    idx_last = len(fp_list) - 1

    x_batch = []
    for idx, fp_image in enumerate(fp_list):
        if np.mod(idx, 100) == 0:
            logging.info('Progress: {0}'.format(idx))
        
        # Load image.
        img = image.load_img(fp_image)
        x = image.img_to_array(img)
        x = tf.cast(x, tf.float32)
        x = center_crop(x, (img_size, img_size))
        
        # Add to batch.
        x_batch.append(x)

        # Predict in batches.
        if len(x_batch) == batch_size or idx == idx_last:
            x_batch = tf.stack(x_batch, axis=0)
            x_batch = preprocess_input(x_batch)
            y_batch = model.predict(x_batch)
            model_features.append(y_batch)
            x_batch = []
        
    model_features = tf.concat(model_features, 0)

    return model_features.numpy()
