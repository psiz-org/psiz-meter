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
"""Databases module.

Functions:
    load_db:
    save_db:
    is_match:
    find:
    update_one:

"""

import pandas as pd

from psiz_meter.database.save_db import save_db


def create_empty_db(fp, columns=['arch_id', 'input_id']):
    """Create empty database.

    Arguments:
        fp: Filepath for database.
        columns (optional): List of column names. The default column
            names constitute the minimum unique identifiers of a model.
            This list does not need to be exhaustive since columns can
            be retroactively added to a pd.DataFrame.

    """
    df = pd.DataFrame(
        columns=columns
    )
    save_db(df, fp)
