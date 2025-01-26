import matplotlib
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt

from procastro.misc import functions
from procastro.parents.calib import CalibBase
from procastro.logging import io_logger
import procastro as pa

__all__ = ['WavSol']

from procastro.misc.functions import use_function, GenNorm, MultiGenNorm


def _print_option(option, grouping):
    return f"{','.join(grouping)}={','.join([str(v) for v in option])}"


def _prepare_mosaic_axes(n, ncols, base=True) -> list[matplotlib.axes.Axes]:
    f = plt.figure()
    gs = f.add_gridspec(ncols=ncols, nrows=int(np.ceil(n / ncols)) + int(base))
    axs = gs.subplots()
    dummy = [ax.remove() for ax in axs[-1]]
    ret = list(axs[:-1].flatten()) + list([f.add_subplot(gs[-1, :])] if base else [])
    for ax in ret[:-2]:
        ax.axis('off')
    return ret


def _extract_around(central, width, out, pix):
    left = central - width / 2
    left_idx = np.argmin(np.abs(left - pix))

    right = left + width / 2
    right_idx = np.argmin(np.abs(right - pix))

    return pix[left_idx: right_idx], out[left_idx: right_idx]


linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


class WavSolSingle:
    def __init__(self,
                 pixwav: Table,
                 function="poly:d4"):
        self.function = function
        self.pixwav = pixwav
        self.mask = None
        self.call_fcn = None

        self.fit_function(function)

        self.xfits = {}
        self.fits = {}
        self.arcs = pa.AstroDir([])

    def __repr__(self):
        return f"<WavSolution with {len(self.pixwav)} lines>"

    def fit_function(self, function,
                     n_sigma=2):
        table = self.pixwav
        pix = table['pix']
        pre_fnc = functions.use_function(function, pix, table['wav'])

        if n_sigma > 0:
            residuals = table['wav'] - pre_fnc(pix)
            std = residuals.data.std()
            mask = np.abs(residuals) < n_sigma*std
            fcn = functions.use_function(function, pix[mask], table['wav'][mask])
        else:
            fcn = pre_fnc
            mask = table['wav'].data * 0 == 0

        self.mask = mask
        self.call_fcn = fcn

        return fcn

    def __call__(self, pix):
        return self.call_fcn(pix)

    def residuals(self,
                  reference="lin_reg",
                  ax=None,
                  alternate_functions=None,
                  legend_title="",
                  plot_kw=None,
                  plot_fcn=None,
                  plot_other=None,
                  ):

        plot_kw = plot_kw or {}
        plot_fcn = plot_fcn or {}
        plot_other = plot_other or {}

        plot_kw = {'ls': '',
                   'marker': 'o',
                   'markersize': 8,
                   } | plot_kw
        plot_fcn = {'ls': '-',
                    'lw': 2,
                    'marker': '',
                    'color': 'black',
                    } | plot_fcn
        plot_other = {'ls': '-',
                      'lw': 1,
                      'marker': '',
                      'color': 'grey',
                      } | plot_other

        if alternate_functions is None:
            alternate_functions = ['poly:d2',
                                   'poly:d3',
                                   f'spline:s{int(0.8*len(self.mask))}'
                                   ]

        if ax is None:
            ax = plt.subplot()

        table = self.pixwav
        pix = table['pix']

        mxpix = max(pix)
        mnpix = min(pix)

        ref = use_function(reference, pix, table['wav'])
        fcn = self.call_fcn
        mask = self.mask
        x = np.linspace(mnpix, mxpix, 40)

        ax.plot(pix[mask], (table['wav'] - ref(pix))[mask],
                zorder=10,
                label="Included lines",
                **plot_kw)
        if mask.sum() != len(mask):
            ax.plot(pix[~mask], (table['wav'] - ref(pix))[~mask],
                    zorder=10,
                    label="Excluded lines",
                    **plot_kw | {'marker': 'X'})
        ax.plot(x, fcn(x) - ref(x),
                zorder=9,
                label=f"{self.function} selected",
                **plot_fcn)

        for linestyle, other_fcn in zip(linestyle_tuple, alternate_functions):
            ax.plot(x, use_function(other_fcn, pix, table['wav'])(x) - ref(x),
                    label=other_fcn,
                    zorder=8,
                    **plot_other | {'ls': linestyle[1]},
                    )
        leg = ax.legend(fontsize=8, title=legend_title,
                        title_fontsize=10)
        leg.set(zorder=15, alpha=0.4)

    def iter_mask_option_arcs(self,
                              arcs: pa.AstroDir | pa.AstroFile,
                              ) -> tuple[np.ndarray, tuple, pa.AstroFile]:
        """
Iterates over all the information that matches column of wavpix and meta of arcs.
        Parameters
        ----------
        arcs

        Returns
        -------
        a (mask, tuple, arc) on each iteration. The mask is for wavpix columns, the option is for each unique arc
        """
        # if arcs is None:
        #     return
        #
        # todo: to_check is not working!!!!

        to_check = [col for col in self.pixwav.colnames if (col in pa.AstroDir(arcs)[0].meta)]

        for arc in arcs:
            option = tuple(arc.values(to_check, single_in_list=True))

            masks = [self.pixwav[col] == val for col, val in zip(to_check, option)]
            mask = np.prod(masks, axis=0)

            yield mask, option, arc

    def add_arc(self,
                arcs: pa.AstroFile | pa.AstroDir,
                ):

        for mask, option, arc in self.iter_mask_option_arcs(arcs):
            self.arcs += arc

    def refit_lines(self,
                    arcs=None,  # only arcs with correct external (typically 'trace') are given.
                    beta=2,
                    width=None,
                    fit_width=40,
                    uncertainty=5,
                    n_sigma=1.5,
                    passes=1,
                    ):
        if arcs is None:
            arcs = self.arcs

        if width is None:
            width = [15, 3, 20]  # values for magellan (guess, min, max)

        # first check the individual centers and widths
        # fit only the elements with the matching column (thinking on elements)
        for mask, option, arc in self.iter_mask_option_arcs(arcs):

            # The following is a mask per-column of interest, it is of interest only where all
            # the columns of interest match
            refit_table = self.pixwav[mask]
            widths = []

            for pix in refit_table['pix']:
                x, y = _extract_around(pix,
                                       fit_width,
                                       arc['pix'],
                                       arc['0'],
                                       )
                orig_center = x[len(x)//2]
                fit = GenNorm(x, y,
                              c=(orig_center,), b=beta, w=width,
                              uncertainty=uncertainty,
                              )

                widths.append(fit.width)

            # now, the supposition is that width can only change linearly with wavelength,
            # so outliers happen because lines are not isolated enough and thus won't be
            # considered to get the width solution. The multi-line fit only uses fixed widths .
            # checking for outliers
            original_centers = refit_table['pix']
            individual_width = np.array(widths)
            mask = np.ones(len(refit_table), dtype=bool)
            for i in range(passes):
                p = np.polyfit(original_centers[mask], individual_width[mask], 1)
                res = individual_width[mask] - np.polyval(p, original_centers[mask])
                sigma = np.std(res)
                mask[mask] = np.abs(res) < n_sigma * sigma
            p = np.polyfit(original_centers[mask], individual_width[mask], 1)

            io_logger.warning(f"Fitted width for option {option} varies"
                              f" between {p[1] + p[0] * original_centers[0]} - "
                              f"{p[1] + p[0] * original_centers[-1]}")

            # widths are fixed for the multi-line fitting
            widths = np.polyval(p, original_centers)
            x = arc['pix']
            if ((original_centers - widths / 2 < x[0]) + (original_centers + widths / 2 > x[-1]))[mask].sum():
                raise ValueError(f"One of the lines in '{self}' is too close to the border, "
                                 f"need to be commented out before")

            fit = MultiGenNorm(x, arc['0'],
                               c=original_centers,
                               w=widths,
                               b=beta,
                               precision_pixel=uncertainty)

            for label, original, new in zip(self.pixwav[['label', 'original_centers']], fit.centers):
                label_mask = np.array(self.pixwav['label'] == label)
                label_idx = np.nonzero(label_mask)[0][0]

                self.pixwav['pix'][label_idx] += new - original

            self.fits[option] = fit
            self.xfits[option] = x

    def plot_fit(self,
                 axs=None,
                 width=30,
                 ):

        if axs is None:
            axs = _prepare_mosaic_axes(len(self.pixwav)+2, ncols=3)

        for ax, (pix, label) in zip(axs, self.pixwav[['pix', 'label']]):

            for af in self.arcs:
                ax.plot(*_extract_around(pix, width,
                                         af.data['pix'],
                                         af.data['0']),
                        label=f"arcs")

            if len(self.fits):
                options = list(self.fits.keys())
                for option in options:
                    ax.plot(*_extract_around(pix, width,
                                             self.fits[option],
                                             self.xfits[option]),
                            label="fit")

            ax.annotate(f"{label}AA", (0.5, 0.05),
                        bbox=dict(boxstyle="round4,pad=.1", fc="white", ec="white", alpha=0.8),
                        ha='center', va='center', xycoords='axes fraction')

        ax = axs[-1]
        for af in self.arcs:
            ax.plot(af.data['pix'],
                    af.data['0'],
                    label=f"arcs ")

        if len(self.fits):
            options = list(self.fits.keys())
            for option in options:
                ax.plot(self.fits[option],
                        self.xfits[option],
                        label="fit")
        ax.legend()

        return axs


####################################
#
# Calibrator
#
####################################

class WavSol(CalibBase):
    def __init__(self,
                 pixwav: "pa.AstroDir | pa.AstroFile",
                 arcs=None,
                 refit=True,
                 group_by='trace',
                 pixwav_function="poly:d4",
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(group_by, list):
            group_by = [group_by]
        self.group_by = group_by

        if isinstance(pixwav, pa.AstroFile):
            pixwav = pa.AstroDir([pixwav])
        if not pixwav[0].spectral:
            raise TypeError("pixwav must have spectral data")

        self.function_name = pixwav_function

        self.wavsols: dict[tuple, WavSolSingle] = {}
        for astrofile in pixwav.mosaic_by(*group_by, in_place=False):
            option = tuple(astrofile.values(*group_by, single_in_list=True))
            self.wavsols[option] = WavSolSingle(astrofile.data, pixwav_function)

        if arcs is None:
            arcs = pa.AstroDir([])
        else:
            if isinstance(arcs, pa.AstroFile):
                arcs = pa.AstroDir([arcs])
            elif not isinstance(arcs, pa.AstroDir):
                raise TypeError("Arcs must be an AstroFile or AstroDir")
            if not arcs[0].spectral:
                raise TypeError("Arcs must have spectral data")

        # arc can potentially be with different option key than pixwav
        self.add_arc(arcs, refit=refit)

    def all_pixs(self):
        ret = []
        for sol in self.wavsols.values():
            ret.extend([tuple(pair) for pair in sol.pixwav[['label', 'wav']] if tuple(pair) not in ret])
        return [x[0] for x in sorted(ret, key=lambda x: x[1])]

    def add_arc(self,
                astrofiles: pa.AstroFile | pa.AstroDir,
                refit=True,
                ):

        for astrofile in pa.AstroDir(astrofiles):
            if not astrofile.spectral:
                raise TypeError("AstroFile must have spectral data")

            # Not every arc might have a wavsols, or viceversa
            option = tuple(astrofile.values(*self.group_by, single_in_list=True))
            if option in self.wavsols:
                self.wavsols[option].add_arc(astrofile)
            else:
                io_logger.warning(f"Ignoring Arc '{astrofile}' whose "
                                  f"'{_print_option(astrofile.values(*self.group_by, single_in_list=True),
                                                    self.group_by)}'"
                                  f" does not have a "
                                  f"matching Wavelength solution.")
                continue

        if refit:
            self.refit()

        return self

    def refit(self,
              beta=2,
              ):
        for option, sol in self.wavsols.items():
            if not len(sol.arcs):
                io_logger.warning(f"There is no arc information for option "
                                  f"'{_print_option(option, self.group_by)}', "
                                  f"keeping original fit")
                continue

            print(sol.arcs)
            sol.refit_lines(sol.arcs, beta=beta)

        return self

    def __repr__(self):
        return (f"<{super().__repr__()} Wavelength Solution. {len(self.wavsols)}x sets of "
                f"{self.group_by}: {list(self.wavsols.keys())}>")

    def __call__(self,
                 data,
                 meta=None,
                 ):
        data, meta = super().__call__(data, meta=meta)

        group_key = tuple([meta[val] for val in self.group_by])
        data['wav'] = self.wavsols[group_key](data['pix'])

        return data, meta

    def residuals(self,
                  reference='lin_reg',
                  alternate_functions=None,
                  ncol=2,
                  plot_kw=None,
                  plot_other=None,
                  plot_fcn=None,
                  axs=None,
                  ):
        """

        Parameters
        ----------
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

        n_res = len(self.wavsols)
        if axs is None:
            f, axs = plt.subplots(nrows=int(np.ceil(n_res/ncol)), ncols=ncol,)

        for (option, wavsol), ax in zip(self.wavsols.items(), axs.flatten()):

            wavsol.residuals(reference=reference,
                             ax=ax,
                             legend_title=f"{','.join(self.group_by)}: {','.join([str(op) for op in option])}",
                             plot_kw=plot_kw,
                             plot_other=plot_other,
                             plot_fcn=plot_fcn,
                             alternate_functions=alternate_functions,
                             )

        axs[0][0].set_title(f"Wavelength residuals with respect to {reference} fit")
        for ax in axs[-1]:
            ax.set_xlabel("pixel")

    def fit_function(self, function, pixwav, option=None):

        if option is None:
            option = list(self.wavsols.keys())
        if not isinstance(option, list):
            option = [option]

        for opt in option:
            if opt in self.wavsols:
                io_logger.warning(f"Overwriting an existing function with options {option}"
                                  f"using function {function}.")

            self.wavsols[opt].fit_function(function, pixwav)

    def plot_fit(self, ncols=3, title=""):

        labels = self.all_pixs()
        axs = _prepare_mosaic_axes(len(labels)+2, ncols)

        for option, sol in self.wavsols.items():
            if not (len(sol.arcs) + len(sol.fits)):
                io_logger.warning(f"No arc nor fits information to plot for "
                                  f"{_print_option(option, self.group_by)}")
                continue

            idxs = []
            for lab in sol.pixwav['label']:
                idxs.append(labels.index(lab))

            inax = np.array(axs)[np.array(idxs)]
            sol.plot_fit(axs=inax, )

        axs[ncols//2].set_title(title)
