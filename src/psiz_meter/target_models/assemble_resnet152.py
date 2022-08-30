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
    assemble_

"""

from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model

from psiz_meter.utils.extract_features import extract_features


def assemble_resnet152(fp_list):
    "ResNet152."
    # Settings.
    n_stimuli = 50000
    n_dim = 2048
    img_size = 224

    # Grab model up to last average pool layer.
    model = ResNet152(weights='imagenet', include_top=False, pooling='avg')
    model_features = extract_features(
        fp_list, model, n_stimuli, n_dim, img_size, preprocess_input
    )

    return model_features
