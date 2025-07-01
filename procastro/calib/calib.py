from typing import Any

import astropy.time as apt

__all__ = ['AstroCalib']

import numpy as np
from astropy.table import Table

from procastro.astrofile.astrofile import AstroFile
from procastro.interfaces.astrocalib import IAstroCalib
from procastro.statics import PADataReturn, PAMetaReturn


class AstroCalib(IAstroCalib):
    def __init__(self,
                 group_by: str | list[str] | None = None,
                 **kwargs,
                 ):
        """
        `AstroCalib` objects contain calibrations that can can be applied sequentially to `AstroFile`
        they can be auto-applied by passing it through `add_calib()` method of `AstroFile`, or they can
        receive an `AstroFile` when calling an initialized `AstroCalib` method (`AstroCalib.__call__()` method)

        `AstroCalib` can hold multiple calibration for a mixed dataset (i.e. many dark-frames with different
        exposure times, many flat-fields with different filters, many wavelength solutions for different traces...),
        the keyword group_by is used to specify the keyword that needs to be used for matching the correct
        calibration to each file.

        Parameters
        ----------
        group_by: str | list[str] | None
        keyword that needs to match to select the correct calibration for each file.  If None, it assumes there is
         only one calibration dataset, complains if not.
        kwargs
        """
        if not isinstance(group_by, (list, tuple)) and group_by is not None:
            group_by = (group_by,)
        self.group_by = group_by

        self._datasets: dict[tuple|str, Any] = {}

    def short(self):
        return ""

    def __str__(self):
        return "AstroCalib"

    def __repr__(self):
        return f"{str(self)}"

    def _get_dataset(self, meta, data=None):
        if self.group_by is None:
            if len(self._datasets) != 1:
                raise ValueError(f"group_by was not specified and there are {len(self._datasets)}"
                                 f" datasets in this {str(self)} ")
            return list(self._datasets.values())[0]
        else:
            # check whether the grouping keyword is in one  of the columns in the table. In that case,
            # each row might want to use a different dataset.

            if data is not None and isinstance(data, Table):

                in_cols = [key in data.colnames for key in self.group_by]
                per_row = any(in_cols)

                if per_row:
                    n = len(data)
                    keys_cols = [[d for d in data[k]] if is_in_cols else [meta[k]]*n
                                 for k, is_in_cols in zip(self.group_by, in_cols)]
                    keys = [tuple(key) for key in zip(*keys_cols)]
                    return np.array([self._datasets[key] for key in keys])
                else:
                    return self._datasets[tuple([meta[k] for k in self.group_by])]

            group_key = tuple([meta[val] for val in self.group_by])
            try:
                return self._datasets[group_key]
            except KeyError as m:
                if len(self.group_by) == 1:
                    return self._datasets[group_key[0]]
                raise KeyError(m)

    def __call__(self,
                 data: PADataReturn,
                 meta: PAMetaReturn,
                 ) -> tuple[PADataReturn, PAMetaReturn]:

        if meta is None:
            meta = {}

        meta['history'] = f"processed by '{self.short()}' on {apt.Time.now().isot}"

        if isinstance(data, AstroFile):
            # the following avoids directly applying an AstroCalib to an AstroFile that already
            # has the same AstroCalib self-applied
            if self in data.get_calib():
                raise RecursionError(f"If passed an AstroFile, then this calibration cannot have been "
                                     f"already passed to that AstroFile")
            meta = data.meta | meta
            data = data.data

        return data, meta
