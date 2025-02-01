import numpy as np
from astropy.table import Table

from procastro.astrofile.static_guess import static_guess_spectral_offset
from procastro.parents.calib import CalibBase
from procastro.logging import io_logger

__all__ = ['WavMosaic']


class WavMosaic(CalibBase):

    def __init__(self,
                 offset_dict=None,
                 offset_key='chip',
                 meta=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if offset_dict is None:
            if meta is not None:
                offset_dict = static_guess_spectral_offset(meta)
            else:
                raise ValueError('offset_dict must be provided, or at least meta for guessing offsets')

        self.offset_dict = offset_dict
        self.offset_key = offset_key

    def __str__(self):
        return f"{super().__str__()}-WavMosaic"

    def __repr__(self):
        return f"<{str(self)}: {self.offset_key} x{len(self.offset_dict.keys())}: {list(self.offset_dict.keys())}>"

    def __call__(self, data, meta=None, *args, **kwargs):
        table, meta = super().__call__(data, meta)

        if not isinstance(table, Table):
            raise TypeError('WavMosaic calibration requires spectral type AstroFile')

        if 'pix' not in table.colnames:
            table['pix'] = np.arange(len(table))

        if self.offset_key in table.colnames:
            table['pix'] += np.array([self.offset_dict[off] for off in table[self.offset_key]])
            meta[self.offset_key] = list(set(table[self.offset_key]))
        elif self.offset_key in meta:
            chip = meta[self.offset_key]
            if isinstance(chip, list):
                if len(chip) > 1:
                    raise TypeError(f"Multiple {self.offset_key} are found in meta information, a "
                                    f"'{self.offset_key}' column is required")
                else:
                    chip = chip[0]
            offset = self.offset_dict[chip]
            table['pix'] += offset
            meta[self.offset_key] = [chip]
        else:
            io_logger.warning(f"No {self.offset_key} information found. "
                              f"Using risky 0 offset along dispersion")

        return table, meta
