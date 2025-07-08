import warnings

import numpy as np
from astropy.table import Table
from astropy.table.column import MaskedColumn
from matplotlib import pyplot as plt
from numpy.ma import MaskedArray
from scipy import interpolate
from scipy import optimize

from procastro.calib.wav_solution_single import WavSolSingle
from procastro.interfaces import IAstroFile, IAstroDir
from procastro.misc import functions
from procastro.misc.functions import use_function
from procastro.calib.calib import AstroCalib
from procastro.config import pa_logger
import procastro as pa

__all__ = ['WavSol']

from procastro.statics import prepare_mosaic_axes


####################################
#
# Calibrator
#
####################################

class WavSol(AstroCalib):
    def __init__(self,
                 pixwav: "pa.AstroDir | pa.AstroFile",
                 arcs=None,
                 separate=None,
                 refit=True,
                 group_by='trace',
                 pixwav_function="poly:d4",
                 beta=2,
                 oversample=2,
                 wav_out=None,
                 align_telluric=None,
                 max_pix_separation=3,
                 **kwargs):
        """
        Wavelength solution that can applied to a Spectral AstroFile.

        Parameters
        ----------
        pixwav
        arcs
        separate
        refit
        group_by: str, list[str]
        indicates which of the meta keywords will be used to match the correct calibration.

        pixwav_function
        beta
        oversample
        wav_out
        align_telluric
        max_pix_separation
        kwargs
        """
        super().__init__(group_by=group_by, **kwargs)
        group_by = self.group_by

        self.beta = beta
        self.oversample = oversample
        self.max_pix_separation = max_pix_separation

        if isinstance(pixwav, str):
            pixwav = pa.AstroDir(pixwav)
        if isinstance(pixwav, pa.AstroFile):
            pixwav = pa.AstroDir([pixwav])
        if not pixwav[0].spectral:
            raise TypeError("pixwav must have spectral data")

        self.function_name = pixwav_function

        self._datasets: dict[tuple, WavSolSingle] = {}
        for astrofile in pixwav.mosaic_by(*group_by, in_place=False):
            option = tuple(astrofile.values(*group_by, single_in_list=True))
            self._datasets[option] = WavSolSingle(astrofile, pixwav_function,
                                                  external=option)

        if arcs is None:
            arcs = pa.AstroDir([])
            refit = False
        else:
            if isinstance(arcs, pa.AstroFile):
                arcs = pa.AstroDir([arcs])
            elif not isinstance(arcs, pa.AstroDir):
                raise TypeError("Arcs must be an AstroFile or AstroDir")
            if not arcs[0].spectral:
                raise TypeError("Arcs must have spectral data")

        # arc can potentially be with different option key than pixwav
        self.add_arc(arcs, refit=refit, separate=separate)
        self.target_wav = self.add_target_wav(wav_out)

        self.align_telluric = align_telluric is not None
        self.col_alignment = align_telluric

    def add_arc(self,
                astrofiles: IAstroFile | IAstroDir,
                separate='element',
                refit=True,
                beta=None,
                ):
        if beta is None:
            beta = self.beta
        self.beta = beta

        for astrofile in pa.AstroDir(astrofiles):
            if not astrofile.spectral:
                raise TypeError("AstroFile must have spectral data")

            # Not every arc might have a wavsols, or viceversa
            option = tuple(astrofile.values(*self.group_by, single_in_list=True))
            if option in self._datasets:
                self._datasets[option].add_arc(astrofile, separate=separate)
            else:
                pa_logger.warning(f"Ignoring Arc '{astrofile}' whose "
                                  f"'{_print_option(astrofile.values(*self.group_by, single_in_list=True),
                                                    self.group_by)}'"
                                  f" does not have a "
                                  f"matching Wavelength solution.")
                continue

        if refit:
            self.refit(beta=beta)

        return self

    def refit(self,
              beta=None,
              uncertainty=9,
              ):
        if beta is None:
            beta = self.beta
        for option, sol in self._datasets.items():
            if not len(sol.arcs):
                pa_logger.warning(f"There is no arc information for option "
                                  f"'{_print_option(option, self.group_by)}', "
                                  f"keeping original fit")
                continue

            sol.refit_lines(beta=beta, uncertainty=uncertainty)
            sol.fit_function()

        return self

    def __repr__(self):
        return (f"<{super().__repr__()} Wavelength Solution. {len(self._datasets)}x sets of "
                f"{tuple(self.group_by)}: {",".join([str(x) for x in self._datasets.keys()])}>")

    def add_target_wav(self,
                       wav: np.ndarray | int | None,
                       decimals: int = 2,
                       ):
        if wav is None:
            return

        if isinstance(wav, np.ndarray):
            self.target_wav = wav
        elif isinstance(wav, int):
            avg_delta_wav = []
            min_wav = []
            max_wav = []
            for wavsol in self._datasets.values():
                pw = wavsol.pixwav
                avg_delta_wav.append((max(pw['wav']) - min(pw['wav'])) / (max(pw['pix']) - min(pw['pix'])))
                min_wav.append(min(pw['wav']))
                max_wav.append(max(pw['wav']))

            dwav = np.round(np.mean(avg_delta_wav), decimals=decimals)
            minw = np.min(min_wav)
            maxw = np.max(max_wav)

            between_pix = (maxw - minw) / dwav
            self.target_wav = np.round(minw, decimals) + (np.arange(wav) - (wav - between_pix) / 2) * dwav
            pa_logger.warning(f"Wavelength scale was set to {wav} pixels between {self.target_wav[0]:.2f}"
                              f" and {self.target_wav[-1]:.2f}. Delta: {self.target_wav[1]-self.target_wav[0]}")
        else:
            raise TypeError("Target wav must be an ndarray (an explicit wav_out) or "
                            "int (number of elements)")

        return self.target_wav

    def __call__(self,
                 data: Table,
                 meta=None,
                 ):
        data, meta = super().__call__(data, meta=meta)

        # get the correct calibration dataset for the astrofile
        wavsol = self._get_dataset(meta)
        wav_out = self.target_wav

        # find the channels that have the information (as opposed to the labeling channels)
        if 'infochn' not in meta:
            raise ValueError("'infochn' is a required meta field for spectral data")
            # infochn = [chn for chn in data.colnames if chn not in ['pix', 'wav']]
            # io_logger.warning(f"No explicit information channels were given. We will"
            #                   f" interpolate all these channels: {infochn} ")
        else:
            infochn = meta['infochn']

        # in case this a timeseries, then get the number of epochs
        n_epochs = data[infochn[0]].shape[-1]

        # it is mandatory to have a 'pix' channel, thus create it by simple enumeration if it
        # doesn't exist.
        if 'pix' not in data.colnames:
            pa_logger.warning(f"Pixel column not found, enumerating pixels from 0")
            data['pix'] = np.arange(len(data))

        # get the pixel offset when alignment is required
        offset = offset_by_telluric(wav_out, data[self.col_alignment]) if self.align_telluric else 0
        wav_in = wavsol(data['pix'][:, None] + np.array(offset)[None, :])

        # masking outliers
        bluest_id_line = min(wavsol.pixwav['wav'])
        reddest_id_line = max(wavsol.pixwav['wav'])
        delta_wav = reddest_id_line - bluest_id_line
        lower_than_id = bluest_id_line - self.oversample * delta_wav / 100 < wav_out
        higher_than_id = wav_out < reddest_id_line + self.oversample * delta_wav / 100

        pix_ref = data['pix']
        delta_pixs = pix_ref[1:] - pix_ref[:-1]
        mask_wav = np.array(lower_than_id * higher_than_id, dtype=bool)
        mask = np.zeros(n_epochs, dtype=bool) + mask_wav[:, None]

        large_jump_mask = ~np.sum([(wav_out[:, np.newaxis] > wav_in[pos, :][np.newaxis, :]) *
                                   (wav_out[:, np.newaxis] < wav_in[pos + 1, :][np.newaxis, :])
                                   for pos in np.nonzero(delta_pixs > self.max_pix_separation)[0]],
                                  dtype=bool, axis=0,
                                  )
        mask *= large_jump_mask

        # interpolate every column
        pa_logger.warning("Interpolating wavelengths" +
                          (" after aligning telluric" if self.align_telluric else ""))
        out_table = Table({'wav': wav_out})  # [:, np.newaxis] * np.ones([1, n_epochs])})
        for col in infochn:
            pa_logger.warning(f" - column {col}")
            fcn = functions.use_function("otf_spline:s0", wav_in, data[col], transpose=True)
            out_table[col] = MaskedColumn(fcn(MaskedArray(wav_out, mask=~mask_wav)),
                                          mask=~mask)
        meta['infochn'] = infochn

        meta['WavSol'] = f"{wavsol.short()}."

        return out_table, meta

    def short(self):
        return "WavSol"

    def plot_width(self, ncol=2,
                   axs=None):
        n_res = len(self._datasets)
        if axs is None:
            f, axs = plt.subplots(nrows=int(np.ceil(n_res/ncol)), ncols=ncol,)

        for (option, wavsol), ax in zip(self._datasets.items(), axs.flatten()):
            wavsol.plot_fit_width(ax=ax,
                                  legend_title=f"{','.join(self.group_by)}: {','.join([str(op) for op in option])}",
                                  )

        axs[0][0].set_title(f"Widths per trace")
        for ax in axs[-1]:
            ax.set_xlabel("pixel")

    def plot_residuals(self,
                       reference='lin_reg',
                       alternate_functions=None,
                       ncol=2,
                       plot_kw=None,
                       plot_other=None,
                       plot_fcn=None,
                       axs=None,
                       title="",
                       ):
        """

        Parameters
        ----------
        title
        ncol
        alternate_functions
        reference
        plot_other
        plot_fcn
        plot_kw:
           'extra_percent' at both sides of extremes when plotting
           'ncol' number of columns for mosaic of axes
        axs
        """

        n_res = len(self._datasets)
        if axs is None:
            f, axs = plt.subplots(nrows=int(np.ceil(n_res/ncol)), ncols=ncol,)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message=".*maximal number.*")

            for (option, wavsol), ax in zip(self._datasets.items(), axs.flatten()):
                wavsol.plot_residuals(reference=reference,
                                      ax=ax,
                                      legend_title=f"{','.join(self.group_by)}: {','.join([str(op) for op in option])}"
                                                   r", $\sigma: ${:.1f}",
                                      plot_kw=plot_kw,
                                      plot_other=plot_other,
                                      plot_fcn=plot_fcn,
                                      alternate_functions=alternate_functions,
                                      )

        axs[0][0].set_title(f"Wavelength residuals with respect to {reference} fit")
        axs[0][-1].set_title(title)
        for ax in axs[-1]:
            ax.set_xlabel("pixel")

    def fit_function(self, function, pixwav, option=None):

        if option is None:
            option = list(self._datasets.keys())
        if not isinstance(option, list):
            option = [option]

        for opt in option:
            if opt in self._datasets:
                pa_logger.warning(f"Overwriting an existing function with options {option}"
                                  f"using function {function}.")

            self._datasets[opt].fit_function(function, pixwav)

    def plot_fit(self, ncols=3, title=""):

        ret = []
        for sol in self._datasets.values():
            ret.extend([tuple(pair)
                        for pair in sol.pixwav[['label', 'wav']]
                        if tuple(pair) not in ret])
        labels = [x[0] for x in sorted(ret, key=lambda x: x[1])]
        axs = prepare_mosaic_axes(len(labels) + 2, ncols)

        for option, sol in self._datasets.items():
            if not (len(sol.arcs) + len(sol.fits)):
                pa_logger.warning(f"No arc nor fits information to plot for "
                                  f"{_print_option(option, self.group_by)}")
                continue

            idxs = []
            for lab in sol.pixwav['label']:
                idxs.append(labels.index(lab))
            idxs.append(len(axs)-1)

            inax = np.array(axs)[np.array(idxs)]
            sol.plot_fit(axs=inax, legend_title=f"{{}} for {option}")

        axs[ncols//2].set_title(title)

    def save_in(self, directory, pattern=None):
        for wavsol in self._datasets.values():
            wavsol.write(directory=directory, pattern=pattern)

    @classmethod
    def get_combinators(cls):
        return cls


def _print_option(option, grouping):
    return f"{','.join(grouping)}={','.join([str(v) for v in option])}"


def closer_pixs(x,
                x0,
                function="spline:s0",
                ):

    n = len(x)
    pix = np.arange(n)
    interp = use_function(function, pix, x)
    if not isinstance(x0, (list, tuple)):
        x0 = [x0]

    ret = []
    for xx0 in x0:
        root = optimize.root_scalar(lambda pixx: interp(pixx) - xx0,
                                    x0=0,
                                    x1=n - 1,
                                    )
        ret.append(root.root)

    return ret


def offset_by_telluric(orig_wav: np.ndarray,
                       fluxes,
                       over_sample: int = 20,
                       degree=2,
                       telluric_band=None,
                       telluric_baseline=None,
                       ):
    """" Return the offsets for each of the arrays in owav such that the arrays are aligned by telluric band """

    if telluric_band is None:
        telluric_band = [7593, 7688]
    if telluric_baseline is None:
        telluric_baseline = [7440, 7840]

    left_right = telluric_baseline
    skip = telluric_band

    left, right = closer_pixs(orig_wav, left_right)
    left, right = int(np.floor(left)), int(np.ceil(right))

    window_flux = fluxes[left: right, :]
    window_wav = orig_wav[left: right]
    skip_left, skip_right = closer_pixs(window_wav, skip)
    skip_left, skip_right = int(np.floor(skip_left)), int(np.ceil(skip_right))

    delta = right - left
    mask = np.zeros(delta) == 0
    mask[skip_left:skip_right] = False

    wav = np.linspace(orig_wav[left], orig_wav[right], int(delta * over_sample + 1))
    ret = []
    norm_spec = []

    for flux in window_flux.transpose():
        anorm_flux = np.polyval(np.polyfit(window_wav[mask], flux[mask], degree), window_wav) - flux
        over_flux = interpolate.UnivariateSpline(window_wav, anorm_flux, s=0.0)(wav)
        norm_spec.append(over_flux)

        max_idx = np.argmax(np.correlate(norm_spec[0], over_flux, mode='full'))
        max_val = (max_idx - (len(over_flux) - 1)) / over_sample

        ret.append(max_val)

    return ret

