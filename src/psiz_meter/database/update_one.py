# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Meter. All Rights Reserved.
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
"""Database module.

Functions:
    update_one

"""

import logging

import pandas as pd

from psiz_meter.database.is_match import is_match


def update_one(df, id_data, assoc_data):
    """Update first row that is found.

    Arguments:
        df: DataFrame of fit database.
        id_data: Dictionary of identifying keys.
        assoc_dict: Dictionary of data that should be associated with
            identifiers.

    """
    # Check if correspondig row already exists.
    loc = is_match(df, id_data)

    if len(df[loc]) > 1:
        logging.warning("Multiple rows in DataFrame match update criteria.")

    if df[loc].empty:
        # Create new row and add.
        df_new = pd.DataFrame({**id_data, **assoc_data}, index=[len(df)])
        df = pd.concat([df, df_new], ignore_index=True)
        # Re-sort by identifier keys to keep things tidy.
        df = df.sort_values(list(id_data.keys()))
    else:
        # Apply updates.
        for k, v in assoc_data.items():
            df.loc[loc, k] = v

    return df
