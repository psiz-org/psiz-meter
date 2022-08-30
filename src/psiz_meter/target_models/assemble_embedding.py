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
    assemble_embedding

"""

import psiz_meter.target_models


def assemble_embedding(model_name, fp_stimuli_list):
    """Assemble embeddings of target model."""
    if model_name == 'vgg16':
        target_embedding = psiz_meter.target_models.assemble_vgg16(
            fp_stimuli_list
        )
    
    if model_name == 'vgg19':
        target_embedding = psiz_meter.target_models.assemble_vgg19(
            fp_stimuli_list
        )

    if model_name == 'xception':
        target_embedding = psiz_meter.target_models.assemble_xception(
            fp_stimuli_list
        )

    if model_name == 'inceptionv3':
        target_embedding = psiz_meter.target_models.assemble_inceptionv3(
            fp_stimuli_list
        )

    if model_name == 'resnet50':
        target_embedding = psiz_meter.target_models.assemble_resnet50(
            fp_stimuli_list
        )

    if model_name == 'resnet101':
        target_embedding = psiz_meter.target_models.assemble_resnet101(
            fp_stimuli_list
        )
    
    if model_name == 'resnet152':
        target_embedding = psiz_meter.target_models.assemble_resnet152(
            fp_stimuli_list
        )

    if model_name == 'resnet50v2':
        target_embedding = psiz_meter.target_models.assemble_resnet50v2(
            fp_stimuli_list
        )

    if model_name == 'InceptionResNetV2':
        target_embedding = psiz_meter.target_models.assemble_inception_resnet_v2(
            fp_stimuli_list
        )

    if model_name == 'densenet121':
        target_embedding = psiz_meter.target_models.assemble_densenet121(
            fp_stimuli_list
        )

    if model_name == 'MobileNet':
        target_embedding = psiz_meter.target_models.assemble_mobilenet(
            fp_stimuli_list
        )

    return target_embedding