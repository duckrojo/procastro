from random import random

import numpy as np

from procastro.astrofile.meta import CaseInsensitiveMeta
from procastro.astrofile.multi import AstroFileMulti


class AstroFileTimeSeries(AstroFileMulti):

    def __repr__(self):
        return (f"<TimeSeries {'Spec' if self.spectral else 'Image'} {len(self._data_file)} files "
                f"{self.filename}>")

    def __init__(self,
                 astrofiles: "str | AstroDir | list[AstroFile]",
                 spectral: bool = None,
                 **kwargs):

        super().__init__(astrofiles, spectral, **kwargs)
        self.multi_epoch_channels = None

    def read(self):

        # if one file is given to timeseries, then check if it is multi- or single-epoch
        if len(self.singles) == 1:
            ret = self.singles[0].data
            meta = self.singles[0].meta

            # if 1 dimension, then there is just one epoch in this timeseries and all channels are multi_channels
            if len(ret[ret.colnames[0]].shape) == 1:
                for colname in ret.colnames:
                    ret[colname] = ret[colname][None, :]

            elif len(ret[ret.colnames[0]].shape) > 2:
                raise TypeError(f"too many dimensions for columns in table {ret}")

        # if more than one file is given to timeseries, then each file is assumed to be just one epoch.
        elif len(self.singles) > 1:
            ret = None
            size = None
            colnames = None
            meta = None

            for idx, single in enumerate(self.singles):
                if ((size is not None and size != len(single))
                        or (colnames is not None and colnames != set(single.data.colnames))):
                    raise ValueError("In a timeseries, all files must have the same size and columns")
                size = len(single)

                new_table = single.data
                colnames = set(new_table.colnames)

                if ret is None:
                    ret = new_table.copy()

                    # column of each element must have just 1-dimension (along row) at start
                    for column in colnames:
                        ret[column] = ret[column][None, :]

                else:
                    for column in colnames:
                        ret[column] = np.concatenate(ret[column], new_table[column][None, :])

                # keep meta from first file only
                if meta is None:
                    meta = single.meta
            raise NotImplementedError(f"Multiple files are not yet supported, multi_epoch_channels in particular")

        else:
            raise ValueError(f"Empty files in {self}. this should have not happened")

        self._meta = CaseInsensitiveMeta(meta)
        self._random = random()

        return ret

    @classmethod
    def get_combinators(cls):
        return [(6, cls)]

    @property
    def id_letter(self):
        return "TS"
