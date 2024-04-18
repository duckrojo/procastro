

__all__ = ['AstroCalib']

from typing import Union, Optional

import numpy as np
from astropy.io.fits import Header

import procastro as pa
import warnings

from .logging import io_logger
from .internal_functions import trim_to_python, common_trim_fcn, extract_common


class AstroCalib(object):
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
    bias_header : astropy.io.fits.Header, optional
        Header of the master bias
    flat_header : astropy.io.fits.Header, optional
        Header of the master bias
    auto_trim : str
        Header field name which is used to cut a portion of the data. This
        field should contain the dimensions of the section to be cut.
        (i.e. TRIMSEC = '[1:1992,1:1708]')
    """

    def __init__(self,
                 bias: 'np.ndarray | pa.AstroFile | None' = None,
                 flat: 'np.ndarray | pa.AstroFile | None' = None,
                 bias_header: Header | dict | None = None,
                 flat_header: Header | dict | None = None,
                 auto_trim: Optional[str] = None):

        # Its always created false, if the add_*() has something, then its
        # turned true.
        self.has_bias = self.has_flat = False

        self.bias: Union[np.ndarray, int, float] = 0.0
        self.flat: Union[np.ndarray, int, float] = 1.0
        self.bias_header = bias_header
        self.flat_header = flat_header

        if auto_trim is not None:
            auto_trim = auto_trim.lower()
        self.auto_trim_keyword: str | None = auto_trim

        if bias is not None:
            self.add_bias(bias)
        if flat is not None:
            self.add_flat(flat)

    def copy(self):
        """returs a new version of itself with same values"""
        return AstroCalib(self.bias, self.flat,
                          bias_header=self.bias_header,
                          flat_header=self.flat_header,
                          auto_trim=self.auto_trim_keyword)

    def _add_calib(self, calib, label, default):
        if isinstance(calib, (int, float, np.ndarray)):
            setattr(self, label, calib)
            # following avoids a positive flag
            if calib == default:
                return
        elif isinstance(calib, pa.AstroFile):
            setattr(self, f"{label}_header", calib.read_headers())
            setattr(self, label, calib.reader())
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

    def reduce(self, data, data_trim=None, verbose=True):
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

        flat = self.flat
        bias = self.bias

        in_data = [data]
        if data_trim is None:
            io_logger.warning("Trim info not found on raw frames... using full figure instead")
            trim = [(1, data.shape[0], 1, data.shape[1])]
        else:
            trim = [data_trim]

        for label in ['bias', 'flat']:
            tdata = vars()[label]
            in_data.append(tdata)

            tmp_header = getattr(self, f'{label}_header')
            if tmp_header is not None:
                tmp_header = tmp_header[0]

            if tmp_header is None or self.auto_trim_keyword not in tmp_header.keys():
                if not isinstance(tdata, (int, float)):

                    io_logger.warning(f"Trim info "
                                      f"{'' if self.auto_trim_keyword is None else self.auto_trim_keyword + ' '}"
                                      f"not found on {label} frames..."
                                      f"using full figure instead")
                    label_trim = (1, tdata.shape[0], 1, tdata.shape[1])
                else:
                    label_trim = trim[0]
                trim.append(label_trim)
            else:
                trim.append(trim_to_python(tmp_header[self.auto_trim_keyword.lower()]))

        if len(set(trim)) != 1:
            common_trim = common_trim_fcn(trim)

            out_data = []
            for label, tdata, trim in zip(['data', 'bias', 'flat'], in_data, trim):
                if isinstance(tdata, (int, float)):  # if tdata is bias = 0 or flat = 1.0, dont trim
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
