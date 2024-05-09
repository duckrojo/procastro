#
#
# Copyright (C) 2013 Patricio Rojo
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of version 2 of the GNU General
# Public License as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
#

__all__ = ['sortmanynsp', 'sortmany', 'accept_object_name',
           'user_confdir', 'defaults_confdir',
           ]

from pathlib import Path
from shutil import copy2

import numpy as np
import operator as op
import warnings
import re

procastro_dir = Path("~/.procastrorc/").expanduser()
procastro_dir.mkdir(exist_ok=True)


def defaults_confdir(file: str,
                     ):
    """
Returns full path to file with default configuration.

    Parameters
    ----------
    file: str
       Name of the file for which the default configuration is requested.

    Returns
    -------

    """
    return Path(__file__).parent.joinpath("..", "defaults", file)


def user_confdir(section: str,
                 start_with_default: bool = True,
                 use_directory: bool = False):
    """returns full path for a _file_ configuration

     """
    default_file = defaults_confdir(section)
    full_file = procastro_dir.joinpath(section)
    if use_directory:
        full_file.mkdir(exist_ok=True, parents=True)

    # if this is the first time the file is checked, then copy it to the user directory
    if start_with_default and not full_file.exists() and default_file.exists():
        copy2(default_file, full_file)
    return full_file


def accept_object_name(name1, name2, planet_match=False, binary_match=False):
    """
    Check if two astronomical names are the same, case independently and punctuation independent.
    Binary stars are identified by upper case. Planets are identified from lower case b onwards.

    Parameters
    ----------
    binary_match: bool
        if True, then names must match binary identification (e.g. Gliese81A != gliese81B)
    planet_match: bool
        if True, then names must match planet identification (e.g. ProxCen b != ProxCen c)
    name1 : str
        Name of object1
    name2 : str
        Name of object1

    Returns
    -------
    bool
    """

    def name_items(name: str) -> tuple[str, str, str, str]:
        astro_name_re = re.compile(r"(?:(?:([a-zA-Z][a-zA-Z0-9]*?)-)|([a-zA-Z]+))[-_ ]?(\d*)([A-Z]*)[ _]?([b-z]*)")
        n_alternate_catalogs = 2  # how many catalog name version are matched (2: "Name|NameNumber-")

        items = astro_name_re.match(name).groups()
        catalog = None
        for n in range(n_alternate_catalogs):
            if catalog is None:
                catalog = items[n]

        return catalog, *items[n_alternate_catalogs:]

    catalog1, number1, binary1, planet1 = name_items(name1)
    catalog2, number2, binary2, planet2 = name_items(name2)

    if catalog1.lower() != catalog2.lower() or number1 != number2:
        return False
    if binary_match and binary1 != binary2:
        return False
    if planet_match and planet1 != planet2:
        return False
    return True


def sortmanynsp(*arr):
    """
    Sort many lists following the order of the first.

    Parameters
    ----------
    arr : list
        Lists to sort

    Returns
    -------
    tuple
        Sorted numpy.ndarray
    """
    return [np.array(r) for r in sortmany(*arr)]


def sortmany(*arr, **kwargs):
    """
    Sort many lists following the order of the first. Optionally using a key

    Parameters
    ----------
    arr : list
        Lists to sort
    key : string, optional
        Key used for sorting

    Returns
    -------
    list
        Sorted list
    """

    if 'key' not in kwargs:
        key = lambda x: x
    else:
        key = kwargs['key']

    keyed=[key(a) for a in arr[0]]

    tups = list(zip(keyed, *arr))
    tups.sort(key=op.itemgetter(0))
    pret = list(zip(*tups))
    return pret[1:]
