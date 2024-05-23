"""
Utilities module for the MLFlow-Hydra Experimentation Framework

    MLFlow Hydra Experimentation Framework
    Copyright (C) 2024 Cristian Cardellino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from collections.abc import MutableMapping
from typing import Any


def _flatten_dict_gen(d: MutableMapping, parent_key: str, sep: str):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep).items()
        elif isinstance(v, list) or isinstance(v, list):
            #  For lists we transform them into strings with a join
            yield new_key, "#".join(map(str, v))
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> dict[str, Any]:
    """
    Flattens a dictionary using recursion (via an auxiliary function
    _flatten_dict_gen). The list/tuples values are flattened as a string.

    Parameters
    ----------
        d: MutableMapping
            Dictionary (or, more generally something that is a MutableMapping) to flatten.
            It might be nested, thus the function will traverse it to flatten it.
        parent_key: str
            Key of the parent dictionary in order to append to the path of keys.
        sep: str
            Separator to use in order to represent nested structures.

    Returns
    -------
        dict[str, Any]
            The flattened dict where each nested dictionary is expressed as a path with
            the `sep`.

    >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': {'e': {'f': 3}}})
    {'a.b': 1, 'a.c': 2, 'd.e.f': 3}
    >>> flatten_dict({'a': {'b': [1, 2]}})
    {'a.b': '1#2'}
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))
