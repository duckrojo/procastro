#
# dataproc - general data processing routines
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


__all__ = ['PrintDebug']


class _printdebug():
    def __init__(self, verblevel=0):
        self.setv(verblevel)
        pass

    def setv(self, verblevel):
        self.verblevel=verblevel

    def __call__(self, string, val=1):
        if (val<=self.verblevel):
            print("DEBUG(", end="")
            print(string, end="")
            print(")")


PrintDebug = _printdebug()
