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
"""Script for reproducing Table 2 in Roads & Love, 2021 CVPR."""

import logging
from pathlib import Path

import psiz_meter.utils
import psiz_meter.database as db


def cvpr2021_table2(fp_project):
    """Run script."""
    # Settings.
    fp_db = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'dbs', 'db_cvpr2021.txt'
    )
    fp_table = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'tables', 'cvpr2021_table2.tex'
    )

    # Version information.
    # NOTE The version v0.1 corresponds to active learning round 118. Likewise
    # v0.2 corresponds to active learning round 195.
    version_round_list = [118, 195]
    version_pretty_list = ['Seed (v0.1)', 'Full (v0.2)']

    # Evaluation settings.
    model_name_list = [
        'psiz', 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152',
        'resnet50v2', 'xception', 'densenet121', 'inceptionv3',
        'InceptionResNetV2', 'MobileNet', 'dc-vgg16',
    ]
    model_pretty_list = [
        'Psych. Emb.', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
        'ResNet50V2', 'Xception', 'DenseNet121', 'InceptionV3',
        'InceptionResNetV2', 'MobileNet', 'DeepCluster'
    ]
    distance_list = ['l1', 'l2', 'cos']
    distance_pretty_list = ['L1', 'L2', 'cos']

    df_results = db.load_db(fp_db)

    # Format and save triplet accuracy results (Table 2 of CVPR 2021 paper).
    caption = (
        'Triplet accuracy for various target models. Triplet accuracy '
        'measures the ability of a target model to correctly predict implicit '
        'triplet inequalities derived from the 8-rank-2 similarity judgments. '
        'The best performing result for each model is emphasized in bold. The '
        'high triplet accuracy of the psychological embedding demonstrates '
        'that there is substantial room for improvement.'
    )
    table_str = psiz_meter.utils.print_triplet_accuracy(
        df_results, model_name_list, version_round_list, distance_list,
        model_pretty_list, version_pretty_list, distance_pretty_list, 
        caption
    )
    f = open(fp_table, 'w')
    f.write(table_str)
    f.close()
    logging.info('LaTeX table saved.')


if __name__ == "__main__":
    # Settings.
    # NOTE: Adjust the filepaths based on your setup.
    fp_project = Path.home() / Path('packages', 'psiz-meter')
    logging.basicConfig(level=logging.INFO)

    cvpr2021_table2(fp_project)
