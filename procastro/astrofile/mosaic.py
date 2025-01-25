from random import random

from astropy.table import Table, vstack

__all__ = ['AstroFileMosaic']

from .multi_files import AstroFileMulti


class AstroFileMosaic(AstroFileMulti):
    def __repr__(self):
        return (f"<Mosaic {'Spec' if self.spectral else 'Image'} {len(self._data_file)} files "
                f"{self.filename}>")

    def __init__(self,
                 astrofiles: "str | AstroDir | list[AstroFile]",
                 spectral: bool = None, **kwargs):

        super().__init__(astrofiles, spectral, **kwargs)

    def read(self):
        ret = Table()

        for idx, single in enumerate(self.singles):

            new_table = single.data
            new_table.meta |= single.meta
            ret = vstack([ret, new_table])

        self._meta = ret.meta
        self._random = random()

        return ret
