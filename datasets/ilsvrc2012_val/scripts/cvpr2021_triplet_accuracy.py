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
"""Script for reproducing values in Table 2 in Roads & Love, 2021 CVPR."""

import logging
import os
from pathlib import Path
import pickle

import numpy as np
import psiz
import tensorflow as tf

import psiz_meter.utils
import psiz_meter.database as db
from psiz_meter.meters import triplet_accuracy

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cvpr2021_triplet_accuracy(fp_project):
    """Run script."""
    # Settings.
    fp_db = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'dbs', 'db_cvpr2021.txt'
    )
    fp_psiz_models = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'psiz_models'
    ) 
    fp_target_emb = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'target_embeddings'
    )
    fp_triplets = fp_project / Path(
        'datasets', 'ilsvrc2012_val', 'triplets'
    )

    # Version information.
    # NOTE The version v0.1 corresponds to active learning round 118. Likewise
    # v0.2 corresponds to active learning round 195.
    version_round_list = [118, 195]
    version_dataset_list = ['ilsvrc_val_v0_1', 'ilsvrc_val_v0_2']

    # Initialize database for results.
    if not fp_db.exists():
        db.create_empty_db(
            fp_db, columns=[
                'model', 'input_id', 'metric', 'distance', 'score'
            ]
        )
    
    # Check triplets directory exists.
    if not os.path.exists(fp_triplets):
        fp_triplets.mkdir(parents=True, exist_ok=True)

    # Convert 8-rank-2 observations to implied triplets for different dataset
    # versions and save. Triplets will be used determine triplet accuracy.
    for version_dataset, version_round in zip(version_dataset_list, version_round_list):
        fp_triplets_version = fp_triplets / Path(
            'obs_{0}.hdf5'.format(version_round)
        )
        # Only derive and save if the file does not already exist.
        if fp_triplets_version.exists():
            logging.info('Triplets {0}: EXISTS'.format(version_dataset))
        else:
            obs, _ = psiz.datasets.load_dataset(version_dataset)
            triplets = psiz_meter.utils.rank_to_triplets(obs)
            triplets.save(fp_triplets_version)
            logging.info('Triplets {0}: SAVED'.format(version_dataset))

    # Evaluation settings.
    model_name_list = [
        'psiz', 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152',
        'resnet50v2', 'xception', 'densenet121', 'inceptionv3',
        'InceptionResNetV2', 'MobileNet', 'dc-vgg16',
    ]
    distance_list = ['l1', 'l2', 'cos']

    # Database settings.
    metric = 'triplet'

    for version_dataset, version_round in zip(version_dataset_list, version_round_list):
        # Load appropriate triplet observations.
        fp_triplets_version = fp_triplets / Path(
            'obs_{0}.hdf5'.format(version_round)
        )
        triplets = psiz.trials.load_trials(fp_triplets_version)

        # Define appropriate (ensemble) PsiZ model.
        fp_model_list = psiz_meter.utils.cvpr2021_model_list(
            version_round, fp_psiz_models
        )

        # Compute triplet accuracy.
        for model_name in model_name_list:
            if model_name == 'psiz':
                # Evaluate triplet accuracy of PsiZ model.
                # NOTE: We do not expect 100% because between- and within-
                # participant agreement can be noisy.
                z_list = []
                for fp_model in fp_model_list:
                    # Load PsiZ embedding.
                    model = tf.keras.models.load_model(fp_model)
                    # Grab mode of embedding posterior.
                    z_list.append(model.stimuli.embeddings.mode().numpy())
                # NOTE: In CVPR 2021 paper, PsiZ models are inferred assuming
                # L1 distance, so it doesn't make sense to evaluate using
                # alternative distances.
                distance_list_override = ['l2']
            else:
                # Evaluate target model.
                fp_ext_embedding_features = fp_target_emb / Path(
                    'emb_{0}.p'.format(model_name)
                )
                # Load embedding(s) of external model.
                z = pickle.load(open(fp_ext_embedding_features, 'rb'))
                # Add placeholder embedding point since triplet observations
                # are indexed assuming `mask_zero=True`, i.e., index `0` is a
                # placeholder.
                z_placeholder = np.zeros([1, z.shape[1]])
                z = np.concatenate([z_placeholder, z], axis=0)
                z_list = [z]
                distance_list_override = distance_list

            for distance in distance_list_override:
                # Evaluate accuracy of (ensemble) model on predicting triplets.
                acc_list = []
                for z in z_list:
                    acc_list.append(
                        triplet_accuracy(z, triplets, distance=distance)
                    )
                acc = np.mean(acc_list)

                # Update database.
                # ID data.
                id_data = {
                    'model': model_name, 'input_id': version_round,
                    'distance': distance, 'metric': metric
                }
                # Associated data.
                assoc_data = {'score': acc}
                df_results = db.load_db(fp_db)
                df_results = db.update_one(df_results, id_data, assoc_data)
                db.save_db(df_results, fp_db)
                logging.info('DB Updated')


if __name__ == "__main__":
    # Settings.
    # NOTE: Adjust the filepaths based on your setup.
    fp_project = Path.home() / Path('packages', 'psiz-meter')
    logging.basicConfig(level=logging.INFO)

    cvpr2021_triplet_accuracy(fp_project)
