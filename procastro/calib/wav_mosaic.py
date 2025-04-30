import numpy as np
from astropy.table import Table

from procastro.astrofile.static_guess import spectral_offset
from procastro.calib.calib import AstroCalib

__all__ = ['WavMosaic']


class WavMosaic(AstroCalib):

    def __init__(self,
                 offset_dict=None,
                 group_by='chip',
                 meta=None,
                 **kwargs):
        super().__init__(group_by=group_by, **kwargs)

        if offset_dict is None:
            if meta is not None:
                offset_dict = spectral_offset(meta)
            else:
                raise ValueError('offset_dict must be provided, or at least meta for guessing offsets')

        self._datasets = offset_dict

    def __str__(self):
        return f"{super().__str__()}-WavMosaic"

    def __repr__(self):
        return f"<{str(self)}: {self.group_by} x{len(self._datasets.keys())}: {list(self._datasets.keys())}>"

    def __call__(self, data, meta=None, *args, **kwargs):
        table, meta = super().__call__(data, meta)

        if not isinstance(table, Table):
            raise TypeError('WavMosaic calibration requires spectral type AstroFile')

        table['pix'] += self._get_dataset(meta, data=data)

        try:
            group_key = list(set(table[self.group_by])) if self.group_by in table.colnames else meta[self.group_by]
        except KeyError:
            group_key = meta[self.group_by[0]]

        meta['history'] = f"using mosaic offsets according to {self.group_by}: {group_key}"
        meta['mosaicby'] = list(self.group_by)

        return table, meta

    def short(self):
        return "WavMosaic"
