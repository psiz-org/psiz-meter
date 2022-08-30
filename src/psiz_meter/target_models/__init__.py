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
"""Target models initialization file."""

from psiz_meter.target_models.assemble_densenet121 import assemble_densenet121
from psiz_meter.target_models.assemble_embedding import assemble_embedding
from psiz_meter.target_models.assemble_inception_resnet_v2 import assemble_inception_resnet_v2
from psiz_meter.target_models.assemble_inceptionv3 import assemble_inceptionv3
from psiz_meter.target_models.assemble_mobilenet import assemble_mobilenet
from psiz_meter.target_models.assemble_resnet50 import assemble_resnet50
from psiz_meter.target_models.assemble_resnet50v2 import assemble_resnet50v2
from psiz_meter.target_models.assemble_resnet101 import assemble_resnet101
from psiz_meter.target_models.assemble_resnet152 import assemble_resnet152
from psiz_meter.target_models.assemble_vgg16 import assemble_vgg16
from psiz_meter.target_models.assemble_vgg19 import assemble_vgg19
from psiz_meter.target_models.assemble_xception import assemble_xception

__all__ = [
    'assemble_densenet121',
    'assemble_embedding',
    'assemble_inception_resnet_v2',
    'assemble_inceptionv3',
    'assemble_mobilenet',
    'assemble_resnet50',
    'assemble_resnet50v2',
    'assemble_resnet101',
    'assemble_resnet152',
    'assemble_vgg16',
    'assemble_vgg19',
    'assemble_xception'
]
