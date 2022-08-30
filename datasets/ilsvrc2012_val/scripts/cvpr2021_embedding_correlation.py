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
"""Script for reproducing values of Table 3 in Roads & Love, 2021 CVPR."""

import logging
import os
from pathlib import Path
import pickle

import numpy as np
import scipy
import tensorflow as tf

import psiz_meter.database as db
import psiz_meter.meters
import psiz_meter.utils

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cvpr2021_table3(fp_project):
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

    # Version information.
    # NOTE The version v0.1 corresponds to active learning round 118. Likewise
    # v0.2 corresponds to active learning round 195.
    version_round_list = [118, 195]

    # Initialize database for results.
    if not fp_db.exists():
        db.create_empty_db(
            fp_db, columns=[
                'model', 'input_id', 'metric', 'distance', 'score'
            ]
        )

    # Evaluation settings.
    model_name_list = [
        'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152',
        'resnet50v2', 'xception', 'densenet121', 'inceptionv3',
        'InceptionResNetV2', 'MobileNet', 'dc-vgg16',
    ]
    sim_list = ['dot', 'cosine']

    # Database settings.
    metric = 'correlation'

    # Compute embedding correlation.
    # For each version, we load the ensemble psychological embedding and
    # compute the (unrolled) similarity matrix.
    for version_round in version_round_list:
        fp_psiz_model_list = psiz_meter.utils.cvpr2021_model_list(
            version_round, fp_psiz_models
        )

        ds_pairs = psiz_meter.utils.pairs_for_ilsvrc2012_val(version_round)

        # Assemble PsiZ ensemble similarity matrix.
        s_psiz, within_rho = psiz_meter.meters.ensemble_marginalized_similarity(
            fp_psiz_model_list, ds_pairs, compute_within=True, mode='spearman',
            verbose=1
        )
        s_psiz = s_psiz.numpy()
        logging.info('Within rho_s: {0:.2f}'.format(within_rho))

        # For each target model, we evaluate the correlation between the
        # psychological embedding and the target model.
        for model_name in model_name_list:

            fp_ext_embedding_features = fp_target_emb / Path(
                'emb_{0}.p'.format(model_name)
            )
            # Load embeddings of external model and compute similarities.
            z = pickle.load(open(fp_ext_embedding_features, 'rb'))
            # Add placeholder embedding point for index "0".
            z_placeholder = np.zeros([1, z.shape[1]])
            z = np.concatenate([z_placeholder, z], axis=0)

            for similarity in sim_list:
                s_target = psiz_meter.meters.basic_similarity(
                    tf.constant(z), ds_pairs, similarity=similarity,
                    verbose=1
                )

                # Evaluate correlation between psiz embeddings and embedding
                # from target model.
                rho, _ = scipy.stats.spearmanr(s_psiz, s_target)

                # Update database.
                # ID data.
                id_data = {
                    'model': model_name, 'input_id': version_round,
                    'distance': similarity, 'metric': metric
                }
                # Associated data.
                assoc_data = {'score': rho}
                df_results = db.load_db(fp_db)
                df_results = db.update_one(df_results, id_data, assoc_data)
                db.save_db(df_results, fp_db)
                logging.info('DB Updated')


if __name__ == "__main__":
    # Settings.
    # NOTE: Adjust the filepaths based on your setup.
    fp_project = Path.home() / Path('packages', 'psiz-meter')
    logging.basicConfig(level=logging.INFO)

    cvpr2021_table3(fp_project)
