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
    is_match

"""


def is_match(df, id_dict):
    """Find a rows based on exact match to identifiers.

    Arguments:
        df:
        id_dict: An identifier dictionary.

    Returns:
        loc: A Boolean index.

    """
    loc = None
    if len(df) > 0:
        for k, v in id_dict.items():
            if loc is None:
                loc = df[k] == v
            else:
                loc = loc & (df[k] == v)
    else:
        loc = []
    return loc
