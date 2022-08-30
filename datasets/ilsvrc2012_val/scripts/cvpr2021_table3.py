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
"""Script for reproducing Table 3 in Roads & Love, 2021 CVPR."""

import logging
from pathlib import Path

import psiz_meter.database as db
import psiz_meter.utils


def cvpr2021_table3(fp_project):
    """Run script."""
    # Settings.
    fp_db = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'dbs', 'db_cvpr2021.txt'
    )
    fp_table = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'tables', 'cvpr2021_table3.tex'
    )

    # Version information.
    # NOTE The version v0.1 corresponds to active learning round 118. Likewise
    # v0.2 corresponds to active learning round 195.
    version_round_list = [118, 195]
    version_pretty_list = ['Seed (v0.1)', 'Full (v0.2)']

    # Evaluation settings.
    model_name_list = [
        'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152',
        'resnet50v2', 'xception', 'densenet121', 'inceptionv3',
        'InceptionResNetV2', 'MobileNet', 'dc-vgg16',
    ]
    model_pretty_list = [
        'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
        'ResNet50V2', 'Xception', 'DenseNet121', 'InceptionV3',
        'InceptionResNetV2', 'MobileNet', 'DeepCluster'
    ]
    sim_list = ['dot', 'cosine']
    sim_pretty_list = ['dot', 'cos']

    df_results = db.load_db(fp_db)

    # Format and save Table 3 of CVPR 2021 paper.
    caption = (
        'The psychological embedding correlation for various target models. '
        'Embedding correlation computes the Spearman correlation between the '
        'similarity matrices of an ensemble of psychological embeddings and a '
        'target model.'
    )
    table_str = psiz_meter.utils.print_embedding_correlation(
        df_results, model_name_list, version_round_list, sim_list,
        model_pretty_list, version_pretty_list, sim_pretty_list,
        caption
    )
    f = open(fp_table, 'w')
    f.write(table_str)
    f.close()
    logging.info('LaTeX Table saved.')


if __name__ == "__main__":
    # Settings.
    # NOTE: Adjust the filepaths based on your setup.
    fp_project = Path.home() / Path('packages', 'psiz-meter')
    logging.basicConfig(level=logging.INFO)

    cvpr2021_table3(fp_project)
