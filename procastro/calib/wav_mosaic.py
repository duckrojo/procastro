import numpy as np
from astropy.table import Table

from procastro.calib.calib import AstroCalib

__all__ = ['WavMosaic']

def spectral_offset() -> dict:
    """The idea is for this function to guess the instrument after reading the meta information."""

    offset = {'imacs': {4: 0,
                        7: 35 + 2048,
                        3: 0,
                        8: 35 + 2048,
                        2: 0,
                        6: 35 + 2048,
                        1: 0,
                        5: 35 + 2048,
                        },
              }

    return {(inst, chip): off
            for inst, val in offset.items()
            for chip, off in val.items()
            }

class WavMosaic(AstroCalib):

    def __init__(self,
                 group_by=('instrument', 'chip'),
                 **kwargs):
        super().__init__(group_by=group_by, **kwargs)


        self._datasets = spectral_offset()

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
