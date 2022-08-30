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
"""Script for reproducing Tables 2 & 3 in Roads & Love, 2021 CVPR.

Tables show results of comparing embeddings trained using ImageNet-HSJ 
and corresponding representations of various target models.

@InProceedings{Roads_Love_2021:CVPR,
    title     = {Enriching ImageNet with Human Similarity Judgments and Psychological Embeddings},
    author    = {Brett D. Roads and Bradley C. Love},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    month     = {6},
    pages     = {3547--3557}
    doi       = {10.1109/CVPR46437.2021.00355},
}

"""

import logging
import os
from pathlib import Path
import pickle

import psiz

import psiz_meter.utils
import psiz_meter.database as db

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def ilsvrc_val_target_embeddings(fp_project, fp_ilsvrc2012_val):
    """Run script."""
    # Settings.
    fp_target_emb = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'target_embeddings'
    )
    version_dataset = 'ilsvrc_val_v0_2'

    # Precompute and save target model feature representations
    # (i.e., embeddings).
    model_name_list = [
        'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152',
        'resnet50v2', 'xception', 'densenet121', 'inceptionv3',
        'InceptionResNetV2', 'MobileNet'
    ]
    # NOTE: Computing feature activations (i.e., the target embedding)
    # requires that you have access to the ILSVRC 2012 dataset, this is not
    # provided by this repository.
    # NOTE: The order of the stimuli in the PsiZ embeddings for the ILSVRC
    # 2012 validation set are in an unusual order, so we load the
    # corresponding catalog to make sure the target model representations are
    # in the same order.
    _, catalog = psiz.datasets.load_dataset(version_dataset)
    catalog.common_path = fp_ilsvrc2012_val
    fp_stimuli_list = catalog.filepath()
    for model_name in model_name_list:
        # Only assemble features if file does not already exist.
        fp_target_embedding = fp_target_emb / Path(
            'emb_{0}.p'.format(model_name)
        )
        if fp_target_embedding.exists():
            logging.info('Embedding {0}: EXISTS'.format(model_name))
        else:
            target_embedding = psiz_meter.target_models.assemble_embedding(
                model_name, fp_stimuli_list
            )
            # Save target embedding.
            pickle.dump(target_embedding, open(fp_target_embedding, 'wb'))
            logging.info('Embedding {0}: SAVED'.format(model_name))


if __name__ == "__main__":
    # Settings.
    # NOTE: Adjust the filepaths based on your setup.
    fp_project = Path.home() / Path('packages', 'psiz-meter')
    fp_ilsvrc2012_val = Path('/fast-data/datasets/ILSVRC/2012/clsloc/val')
    logging.basicConfig(level=logging.INFO)

    ilsvrc_val_target_embeddings(fp_project, fp_ilsvrc2012_val)
