import numpy as np
from astropy.table import Table

from procastro.core.astrofile.static_guess import static_guess_spectral_offset
from procastro.calib import CalibBase
from procastro.core.logging import io_logger

__all__ = ['WavMosaic']


class WavMosaic(CalibBase):

    def __init__(self, offset_dict=None, offset_key='chip', meta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if offset_dict is None:
            if meta is not None:
                offset_dict = static_guess_spectral_offset(meta)
            else:
                raise ValueError('offset_dict must be provided, or at least meta for guessing offsets')

        self.offset_dict = offset_dict
        self.offset_key = offset_key

    def __call__(self, data, meta=None, *args, **kwargs):
        table, meta = super().__call__(data, meta)

        if not isinstance(table, Table):
            raise TypeError('WavMosaic calibration requires spectral type AstroFile')

        if 'pix' not in table.colnames:
            table['pix'] = np.arange(len(table))

        if self.offset_key in meta:
            chip = str(meta[self.offset_key])
            offset = self.offset_dict[chip]
            table['pix'] += offset
        elif self.offset_key in table.colnames:
            table['pix'] += np.array([self.offset_dict[off] for off in table[self.offset_key]])
        else:
            io_logger.warning(f"No {self.offset_key} information found. "
                              f"Using risky 0 offset along dispersion")
