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
    print_triplet_accuracy

"""

import numpy as np


def print_triplet_accuracy(
        df_results, model_name_list, version_round_list, distance_list,
        model_pretty_list, version_pretty_list, distance_pretty_list,
        caption):
    """Output LaTeX table for triplet accuracy.
    
    Returns:
        complete_table_str: The complete LaTeX table string.

    """
    complete_table_str = ''
    n_distance = len(distance_list)

    df_sub = df_results[df_results.metric == 'triplet'].copy()
    df_sub['score'] = df_sub['score'] * 100

    # Print table header.
    complete_table_str += '\\begin{table}' + '\n'
    complete_table_str += '\\begin{center}' + '\n'
    tabular_str = '\\begin{tabular}{|l|'
    heading_labels_str = 'Target Model'
    subheading_labels_str = ' '
    for v in version_pretty_list:
        heading_labels_str += ' & \\multicolumn{' + str(n_distance) + '}'
        heading_labels_str += '{|c|}{\\bf{' + v + '}}'
        for d in distance_pretty_list:
            tabular_str += 'c|'
            subheading_labels_str += ' & ' + d
    tabular_str += '}'
    heading_labels_str += ' \\\\'
    subheading_labels_str += ' \\\\'
    complete_table_str += tabular_str + '\n'
    complete_table_str += '\\hline' + '\n'
    complete_table_str += heading_labels_str + '\n'
    complete_table_str += subheading_labels_str + '\n'
    complete_table_str += '\\hline\\hline' + '\n'

    # Print data.
    for idxm, m in enumerate(model_name_list):
        df_mod = df_sub[df_sub['model'] == m]
        if model_pretty_list[idxm] == 'DeepCluster':
            complete_table_str += 'DeepCluster & & & & & & \\\\' + '\n'
            row_str = '\\hspace{3mm}VGG16 '
        elif model_pretty_list[idxm] == 'InceptionResNetV2':
            complete_table_str += 'Inception & & & & & & \\\\' + '\n'
            row_str = '\\hspace{3mm}ResNetV2 '
        else:
            row_str = '{0} '.format(model_pretty_list[idxm])
        for v in version_round_list:
            df_ver = df_mod[df_mod['input_id'] == v]
            acc_list = []
            for d in distance_list:
                df_acc = df_ver[df_ver['distance'] == d]
                if len(df_acc) > 0:
                    acc_list.append(df_acc['score'].values[0])
                else:
                    acc_list.append(-np.inf)
            # Identify largest.
            idx_largest = np.argmax(acc_list)
            for idx, acc in enumerate(acc_list):
                if acc == -np.inf:
                    row_str += '& -- '
                else:
                    if idx == idx_largest:
                        # Use bold.
                        row_str += '& \\bf{{{0:.1f}}} '.format(acc)
                    else:
                        row_str += '& {0:.1f} '.format(acc)

        row_str += '\\\\'
        complete_table_str += row_str + '\n'

    # Close table.
    complete_table_str += '\\hline' + '\n'
    complete_table_str += '\\end{tabular}' + '\n'
    complete_table_str += '\\end{center}' + '\n'
    complete_table_str += '\\caption{' + caption + '}' + '\n'
    complete_table_str += '\\end{table}' + '\n'

    return complete_table_str
