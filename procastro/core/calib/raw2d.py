

__all__ = ['CalibRaw2D']

from typing import Union, Optional

import numpy as np

import procastro as pa
import warnings

from procastro.core.calib.base import CalibBase
from procastro.core.logging import io_logger
from procastro.core.internal_functions import trim_to_python, common_trim_fcn, extract_common


class CalibRaw2D(CalibBase):
    """
    Object to hold calibration frames.

    Since several AstroFiles might use the same calibration frames, one
    AstroCalib object might be shared by more than one AstroFile. For instance,
    all the files initialized through AstroDir share a single calibration
    object.

    Attributes
    ----------
    has_bias : bool
    has_flat : bool

    Parameters
    ----------
    bias : np.ndarray, AstroFile
        Master Bias
    flat : np.ndarray, AstroFile
        Master Flat
    auto_trim : str
        Header field name which is used to cut a portion of the data. This
        field should contain the dimensions of the section to be cut.
        (i.e. TRIMSEC = '[1:1992,1:1708]')
    """

    def __init__(self,
                 bias: 'pa.AstroFile | None' = None,
                 flat: 'pa.AstroFile | None' = None,
                 auto_trim: bool | str = True,
                 **kwargs):

        # It is always created false, if the add_*() has something, then its
        # turned true.
        super().__init__(**kwargs)

        possible_trims = ['TRIMSEC','DATASEC']

        try:
            meta = bias.meta
        except AttributeError:
            try:
                meta = flat.meta
            except AttributeError:
                meta = {}

        if isinstance(auto_trim, str):
            self.auto_trim_keyword = auto_trim.upper()
        elif auto_trim:
            for k in possible_trims:
                if k.upper() in meta:
                    self.auto_trim_keyword = k.upper()
                    break
            else:
                io_logger.warning("No Trim keyword found in flat or bias")
                self.auto_trim_keyword = None

        if bias is not None:
            self.add_bias(bias)
        if flat is not None:
            self.add_flat(flat)

    def copy(self):
        """returs a new version of itself with same values"""
        return CalibRaw2D(self.bias, self.flat,
                          bias_header=self.bias_header,
                          flat_header=self.flat_header,
                          auto_trim=self.auto_trim_keyword)

    def _add_calib(self, calib, label, default):
        if isinstance(calib, (int, float)):
            setattr(self, label, calib)
            # following avoids a positive flag
            if calib == default:
                return
        elif isinstance(calib, pa.AstroFile):
            setattr(self, label, calib)
        else:
            raise ValueError(f"Master {label} supplied was not recognized.")

        setattr(self, f"has_{label}", True)

    def add_bias(self, bias):
        """
        Add Master Bias to Calib object.

        Parameters
        ----------
        bias : dict indexed by exposure time, array_like or AstroFile
            Master bias to be included

        Raises
        ------
        ValueError
            If the bias type is invalid
        """

        self._add_calib(bias, "bias", 0)
        return self

    def add_flat(self, flat):
        """
        Add master flat to Calib object

        Parameters
        ----------
        flat: dict indexed by filter name, array_like, AstroFile
            Master flat to be included

        Raises
        ------
        ValueError
            If the bias type is invalid
        """
        self._add_calib(flat, "flat", 1)
        return self

    def __call__(self,
                 astrofile,
                 data,
                 data_trim=None,
                 verbose=True,
                 ):
        """
        Process given "data" using the bias and flat contained in this instance

        Parameters
        ----------
        data_trim
        verbose
        data : array_like
            Data to be reduced
        Returns
        -------
        array_like
            Reduced data

        """

        if len(data.shape) != 2:
            return data

        flat = self.flat
        bias = self.bias

        in_data = [data]
        try:
            label_trim = astrofile.meta[self.auto_trim_keyword]
        except KeyError:
            io_logger.warning(f"Trim info "
                              f"{'' if self.auto_trim_keyword is None else self.auto_trim_keyword + ' '}"
                              f"not found on raw frames..."
                              f"using full figure instead")
            label_trim = (1, data.shape[0], 1, data.shape[1])
        trims = [label_trim]

        # a first loop to get trim info
        for frame in [bias, flat]:
            tdata = frame.data
            in_data.append(tdata)

            tmp_header = frame.meta

            if tmp_header is None or self.auto_trim_keyword not in tmp_header.keys():
                if not isinstance(tdata, (int, float)):

                    io_logger.warning(f"Trim info "
                                      f"{'' if self.auto_trim_keyword is None else self.auto_trim_keyword + ' '}"
                                      f"not found on {frame} frames... "
                                      f"using full figure instead")
                    label_trim = (1, tdata.shape[0], 1, tdata.shape[1])
                else:
                    label_trim = trims[0]
                trims.append(label_trim)
            else:
                trims.append(trim_to_python(tmp_header[self.auto_trim_keyword.upper()], maxlims=tdata.shape))

        if len(set(trims)) != 1:
            common_trim = common_trim_fcn(trims)

            out_data = []
            # a second loop to apply common trim
            for label, tdata, trim in zip(['data', 'bias', 'flat'], in_data, trims):
                if isinstance(tdata, (int, float)):  # if tdata is bias = 0 or flat = 1.0, don't trim
                    out_data.append(tdata)
                    trimmed = False
                else:
                    out, trimmed = extract_common(tdata, trim, common_trim)
                    out_data.append(out)

                if trimmed and verbose:
                    io_logger.info(f"Adjusting {label} shape to minimum common trim [{self.auto_trim_keyword}: "
                                   f"({str(trim)}) -> ({str(common_trim)})]")
            data, bias, flat = out_data

        debias = data - bias
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            de_flat = debias / flat

        return de_flat
