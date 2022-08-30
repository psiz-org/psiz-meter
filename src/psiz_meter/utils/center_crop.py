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
    center_crop

"""

import numpy as np
import tensorflow as tf


def center_crop(image, shape, init_shape=None):
    target_width = shape[0]
    target_height = shape[1]
    if init_shape is None:
        init_shape = np.maximum(target_width + 1, target_height + 1)
    else:
        init_shape += 1

    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]
    im = image
    ratio = 0
    if(initial_width < initial_height):
        ratio = tf.cast(init_shape / initial_width, tf.float32)
        h = tf.cast(initial_height, tf.float32) * ratio
        im = tf.image.resize(im, (init_shape, h), method="bicubic")
    else:
        ratio = tf.cast(init_shape / initial_height, tf.float32)
        w = tf.cast(initial_width, tf.float32) * ratio
        im = tf.image.resize(im, (w, init_shape), method="bicubic")
    width = tf.shape(im)[0]
    height = tf.shape(im)[1]
    startx = width//2 - (target_width//2)
    starty = height//2 - (target_height//2)
    im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
    return im
