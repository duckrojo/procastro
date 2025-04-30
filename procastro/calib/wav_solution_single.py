from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from procastro.interfaces import IAstroFile, IAstroDir
from procastro.statics import prepare_mosaic_axes
from procastro.logging import io_logger
from procastro.misc import functions
from procastro.misc.functions import use_function, GenNorm, MultiGenNorm
import procastro as pa


class WavSolSingle:
    def __init__(self,
                 astrofile: IAstroFile,
                 function="poly:d4",
                 external=None,

                 ):
        """

        Parameters
        ----------
        astrofile
        function
        """
        self.std = None
        self.function = function
        self.astrofile = astrofile
        self.pixwav = astrofile.data.copy()

        self.mask = None
        self.call_fcn = None

        self.fit_function(function)

        self.xfits = {}
        self.fits = {}
        self.arcs: IAstroDir = pa.AstroDir([])
        self.fit_lin_width = {}
        self.fit_widths = {}
        self.fit_widths_mask = {}
        self.separate_calib = []
        self.external = external

    def plot_fit_width(self, ax=None, legend_title=""):
        if ax is None:
            ax = plt.subplot()
        for mask, option, arc in self.iter_mask_option_arcs(self.arcs):
            fit_mask = self.fit_widths_mask[option]
            x = self.pixwav[mask]['pix']

            ax.plot(x[fit_mask],
                    self.fit_widths[option][fit_mask], marker='o', color='b', ls='',
                    label=f"included")
            ax.plot(x[~fit_mask],
                    self.fit_widths[option][~fit_mask], marker='X', color='r', ls='',
                    label="excluded")
            ax.plot(x, np.polyval(self.fit_lin_width[option], x),
                    label=f"fitted width for {self.separate_calib}",
                    )
        ax.legend(title=legend_title.format())

    def short(self):
        return f"{len(self.pixwav)}L, {self.std:.2f}S, {self.function}"

    def __repr__(self):
        return (f"<WavSolution with {len(self.pixwav)} lines, {len(self.arcs)} arcs, "
                f"function '{self.function}', std {self.std}>")

    def fit_function(self, function=None,
                     n_sigma=2):
        if function is None:
            function = self.function

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

        self.std = (table['wav'] - pre_fnc(pix)).data[mask].std()
        self.mask = mask
        self.call_fcn = fcn
        self.function = function

        return fcn

    def __call__(self, pix):
        return self.call_fcn(pix)

    def plot_residuals(self,
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
            ax.plot(x, use_function(other_fcn, pix[mask], table['wav'][mask])(x) - ref(x),
                    label=other_fcn,
                    zorder=8,
                    **plot_other | {'ls': linestyle[1]},
                    )

        leg = ax.legend(fontsize=8, title=legend_title.format(self.std),
                        title_fontsize=10, ncols=2)
        leg.set(zorder=15, alpha=0.4)

    def iter_mask_option_arcs(self,
                              arcs: IAstroDir | IAstroFile,
                              ) -> tuple[np.ndarray, tuple, IAstroFile]:
        """
Iterates over all the information that matches column of wavpix and meta of arcs.
        Parameters
        ----------
        arcs

        Returns
        -------
        a (mask, tuple, arc) on each iteration. The mask is for wavpix columns, the option is for each unique arc
        """
        if arcs is None:
            return
        arcs = pa.AstroDir(arcs)

        if self.separate_calib is None:
            separate = [col for col in self.pixwav.colnames if (col in pa.AstroDir(arcs)[0].meta)]
        elif isinstance(self.separate_calib, str):
            separate = [self.separate_calib]
        elif not isinstance(self.separate_calib, list):
            raise TypeError('separate must be string, a list of strings, or None')
        else:
            separate = self.separate_calib

        for arc in arcs:
            option = tuple(arc.values(separate, single_in_list=True))

            if not len(option):
                mask = np.ones(len(self.pixwav), dtype=bool)
            else:
                masks = [self.pixwav[col] == val for col, val in zip(separate, option)]
                mask = np.prod(masks, axis=0, dtype=bool)

            yield mask, option, arc

    def add_arc(self,
                arcs: IAstroFile | IAstroDir,
                separate=None,
                ):
        """

        Parameters
        ----------
        arcs
        separate:
           Key used to separate calibs, otherwise it will merge them. They should be
           both in pixwav columns and in arcs' meta.
           If none, then find all columns that are in both pixwav and arcs.
           If you do not want to separate, use []
        """
        self.separate_calib = separate

        for mask, option, arc in self.iter_mask_option_arcs(arcs):
            if arc not in self.arcs:
                self.arcs += arc

    def refit_lines(self,
                    arcs=None,  # only arcs with correct external (typically 'trace') are given.
                    beta=2,
                    width=None,
                    fit_width=50,
                    uncertainty=9,
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

            arc_table = arc.data
            if 'pix' not in arc_table.colnames:
                arc_table['pix'] = np.arange(len(arc.data))

            for pix in refit_table['pix']:
                x, y = _extract_around(pix,
                                       fit_width,
                                       arc_table['pix'],
                                       arc_table['0'],
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
                new_mask = np.abs(res) < n_sigma * sigma
                mask[mask] = new_mask
            p = np.polyfit(original_centers[mask], individual_width[mask], 1)
            self.fit_lin_width[option] = p
            self.fit_widths[option] = np.array(widths)
            self.fit_widths_mask[option] = mask

            io_logger.warning(f"Fitted line widths for option "
                              f"'{",".join([str(x) 
                                            for x in list(self.external) + list(option)])}'"
                              f" varies between {p[1] + p[0] * original_centers[0]:.1f} - "
                              f"{p[1] + p[0] * original_centers[-1]:.1f}")

            # widths are fixed for the multi-line fitting
            widths = np.polyval(p, original_centers)
            x = arc_table['pix']
            if ((original_centers - widths / 2 < x[0]) + (original_centers + widths / 2 > x[-1]))[mask].sum():
                raise ValueError(f"One of the lines in '{self}' is too close to the border, "
                                 f"need to be commented out before")

            fit = MultiGenNorm(x, arc_table['0'],
                               c=original_centers,
                               w=widths,
                               b=beta,
                               precision_pixel=uncertainty)

            for label, original, new in zip(refit_table['label'],
                                            original_centers, fit.centers):
                label_mask = np.array(self.pixwav['label'] == label)
                label_idx = np.nonzero(label_mask)[0][0]

                self.pixwav['pix'][label_idx] += new - original

            self.fits[option] = fit(x)
            self.xfits[option] = x

    def plot_fit(self,
                 axs=None,
                 width=30,
                 legend_title="",
                 ):

        if axs is None:
            axs = prepare_mosaic_axes(len(self.pixwav) + 2, ncols=3)

        for ax, (pix, label) in zip(axs, self.pixwav[['pix', 'label']]):

            for i, af in enumerate(self.arcs):
                if 'pix' not in af.data.colnames:
                    arc_pix = np.arange(len(af.data['0']))
                else:
                    arc_pix = af.data['pix']
                x, y = _extract_around(pix, width,
                                       arc_pix,
                                       af.data['0'])

                ax.plot(y)

            if len(self.fits):
                options = list(self.fits.keys())
                for option in options:
                    x, y = _extract_around(pix, width,
                                           self.xfits[option],
                                           self.fits[option])

                    ax.plot(y, ls="--")

            ax.annotate(f"{label}AA", (0.5, 0.05),
                        bbox=dict(boxstyle="round4,pad=.1", fc="white", ec="white", alpha=0.8),
                        ha='center', va='center', xycoords='axes fraction')

        ax = axs[-1]
        for i, af in enumerate(self.arcs):
            if 'pix' not in af.data.colnames:
                arc_pix = np.arange(len(af.data['0']))
            else:
                arc_pix = af.data['pix']
            ax.plot(arc_pix,
                    af.data['0'],
                    label=None if i else legend_title.format(f"Arc"))

        if len(self.fits):
            options = list(self.fits.keys())
            for i, option in enumerate(options):
                ax.plot(self.fits[option],
                        self.xfits[option],
                        label=None if i else legend_title.format("Fit"),
                        ls="--")
        ax.legend(fontsize=8)

        return axs

    def write(self, directory=None, pattern=None):
        if pattern is None:
            save_filename = self.astrofile.filename
        else:
            save_filename = pattern.format(**self.astrofile.meta)

        if directory is None:
            filename = save_filename
        else:
            directory = Path(directory)
            if directory.exists() and not directory.is_dir():
                raise FileExistsError(f"Target destination exists and is not a directory: {directory}")
            if not directory.exists():
                directory.mkdir(parents=True)

            filename = directory / Path(save_filename).name

        pa.AstroFile(self.pixwav).write_as(filename)


def _extract_around(central, width, pix, out):
    left = central - width / 2
    left_idx = np.argmin(np.abs(left - pix))

    right = central + width / 2
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
