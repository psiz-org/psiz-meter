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
"""Utility module.

Functions:
    cvpr2021_model_list

"""

from pathlib import Path


def cvpr2021_model_list(version_round, fp_psiz_models):
    """Create list of model filepaths for the requested version.

    Arguments:
        version_round: Integer indicating the desired version round.
        fp_psiz_models: The root directory where the TF model
            directories live.

    Returns:
        fp_psiz_model_list: A list of PsiZ model filepaths that should
            be used to load a complete ensemble.

    """
    if version_round == 118:
        # Seed set (v0.1).
        fp_psiz_model_list = [
            fp_psiz_models / Path('emb-0-118-9-0'),
            fp_psiz_models / Path('emb-0-118-9-1'),
            fp_psiz_models / Path('emb-0-118-9-2')
        ]
    elif version_round == 195:
        # Full validation set (v0.2).
        fp_psiz_model_list = [
            fp_psiz_models / Path('emb-0-195-4-0'),
            fp_psiz_models / Path('emb-0-195-4-1'),
            fp_psiz_models / Path('emb-0-195-4-2'),
        ]
    else:
        raise NotImplementedError('Requested round not implemented.')

    return fp_psiz_model_list

