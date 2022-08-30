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
    find

"""

from psiz_meter.database.is_match import is_match


def find(df, match_dict):
    """Find a rows based on exact match to identifiers.

    Arguments:
        df:
        match_dict: An dictionary of key values to match.

    Returns:
        df: A DataFrame with any matching rows.

    """
    loc = is_match(df, match_dict)
    return df[loc]
