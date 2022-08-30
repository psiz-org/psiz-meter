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
    assemble_vgg16

"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

from psiz_meter.utils.extract_features import extract_features


def assemble_vgg16(fp_list):
    """VGG 16."""
    # Settings.
    n_stimuli = 50000
    n_dim = 4096
    img_size = 224


    base_model = VGG16(weights='imagenet')
    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )
    model_features = extract_features(
        fp_list, model, n_stimuli, n_dim, img_size, preprocess_input
    )
    return model_features
