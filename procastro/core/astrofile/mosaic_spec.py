from random import random

import numpy as np
from astropy.table import Table, vstack

from procastro.core.statics import identity
from procastro.core.astrofile.spec import AstroFileSpec


__all__ = ['AstroFileMosaicSpec']

from procastro.core.astrofile.static_guess import static_guess_spectral_offset
from procastro.core.logging import io_logger


class AstroFileMosaicSpec(AstroFileSpec):
    def __init__(self,
                 astrofiles,
                 offset: dict[str, float] | None = None,
                 offset_key='chip',
                 **kwargs):

        if isinstance(astrofiles, str):
            astrofiles = [astrofiles]

        super().__init__(astrofiles[0], do_not_read=True, **kwargs)

        self._data_file = f"multi x{len(astrofiles)}"
        self.singles = [AstroFileSpec(af, **kwargs) for af in astrofiles]

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

            if self.offset_key not in single:
                io_logger.warning(f"No {self.offset_key} information found in {single}. "
                                  f"Using risky 0 offset along dispersion")
                offset = 0
            else:
                chip = str(single[self.offset_key])
                offset = self.offset_values[chip]
            table['pix'] += offset

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


if __name__ == "__main__":
    import procastro as pa

    name_pattern = "ob{trace:02d}_{chip:d}_arc_{element}.fits"
    sp = pa.AstroFileMosaicSpec(["../../../sample_files/ob06_6_arc_Ar.fits",
                                 "../../../sample_files/ob06_1_arc_Ar.fits",
                                 ],
                                meta_from_name=name_pattern,
                                )
    sp.plot()
