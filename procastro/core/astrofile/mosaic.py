from pathlib import Path
from random import random

import numpy as np
from astropy.table import Table, vstack

from procastro.core.astrofile.static_guess import static_guess_spectral_offset
from procastro.core.logging import io_logger
from procastro.core.statics import identity
from procastro.core.astrodir import AstroDir
from procastro.core.astrofile import AstroFile

__all__ = ['AstroFileMosaic']


class AstroFileMosaic(AstroFile):

    @property
    def filename(self):
        return "(" + ', '.join([Path(df).name for df in self._data_file]) + ")"

    def __repr__(self):
        return (f"<Multi {'Spec' if self.spectral else 'Image'} {len(self._data_file)} files "
                f"{self.filename}>")

    def __new__(cls, *args, **kwargs):
        """
        If passed an AstroFile, then do not create a new instance, just pass
        that one. If passed None, return None
        """
        new_args = len([key for key, val in kwargs.items() if val is not None])
        if args and ((isinstance(args[0], AstroFile) and new_args == 0) or
#                     (isinstance(args[0], AstroDir) and new_args == 0) or
                     args[0] is None):
            return args[0]

        return super().__new__(cls)

    def __init__(self,
                 astrofiles: str | AstroDir | list[AstroFile],
                 offset: dict[str, float] | None = None,
                 offset_key='chip',
                 spectral: bool = None,
                 **kwargs):

        if isinstance(astrofiles, str):
            astrofiles = [astrofiles]

        if spectral is None:
            try:
                spectral = astrofiles[0].spectral
            except AttributeError:
                raise TypeError("If not given AstroFile iterator to AstroFileMosaic, spectral must be specified")
        self.spectral = spectral

        super().__init__(astrofiles[0], spectral=spectral, do_not_read=True, **kwargs)

        self.singles = [AstroFile(af, spectral=spectral, **kwargs) for af in astrofiles]

        self._data_file = tuple([af.filename for af in astrofiles])

        self.offset_key = offset_key.upper()
        self.offset_values = self.update_offset(offset)

        # first read storing in cache
        identity(self.data)

    def update_offset(self, offset):
        if offset is None:
            offset = static_guess_spectral_offset(self.singles[0].meta)

        self.offset_values = {str(k): v for k, v in offset.items()}
        self._random = random()  # forces re-read of cache

        return self.offset_values

    def read(self):
        ret = Table()
        meta = {}
        for idx, single in enumerate(self.singles):
            table = single.data
            if 'pix' not in table.colnames:
                table['pix'] = np.arange(len(table))
            table['astrofile'] = idx

            if self.offset_key in single.meta:
                chip = str(single[self.offset_key])
                offset = self.offset_values[chip]
                table['pix'] += offset
            elif self.offset_key in table.colnames:
                table['pix'] += np.array(self.offset_values)[table[self.offset_key]]
            else:
                io_logger.warning(f"No {self.offset_key} information found in {single}. "
                                  f"Using risky 0 offset along dispersion")

            ret = vstack([ret, single.data])

            meta_of_key = meta[self.offset_key] if self.offset_key in meta else None
            meta |= single.meta

            if meta_of_key is None:
                meta_of_key = []
            elif not isinstance(meta_of_key, list):
                meta_of_key = [meta_of_key]

            meta[self.offset_key] = meta_of_key + [single.meta[self.offset_key]]

        self._meta = meta
        self._random = random()

        return ret
