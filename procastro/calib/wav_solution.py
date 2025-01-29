import numpy as np
from matplotlib import pyplot as plt

from procastro.calib.wav_solution_single import WavSolSingle
from procastro.parents.calib import CalibBase
from procastro.logging import io_logger
import procastro as pa

__all__ = ['WavSol']

from procastro.statics import prepare_mosaic_axes


####################################
#
# Calibrator
#
####################################

class WavSol(CalibBase):
    def __init__(self,
                 pixwav: "pa.AstroDir | pa.AstroFile",
                 arcs=None,
                 separate=None,
                 refit=True,
                 group_by='trace',
                 pixwav_function="poly:d4",
                 beta=2,
                 **kwargs):
        super().__init__(**kwargs)

        self.beta = beta

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
            self.wavsols[option] = WavSolSingle(astrofile, pixwav_function,
                                                external=option)

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
        self.add_arc(arcs, refit=refit, separate=separate)

    def all_pixs(self):
        ret = []
        for sol in self.wavsols.values():
            ret.extend([tuple(pair) for pair in sol.pixwav[['label', 'wav']] if tuple(pair) not in ret])
        return [x[0] for x in sorted(ret, key=lambda x: x[1])]

    def add_arc(self,
                astrofiles: pa.AstroFile | pa.AstroDir,
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
            if option in self.wavsols:
                self.wavsols[option].add_arc(astrofile, separate=separate)
            else:
                io_logger.warning(f"Ignoring Arc '{astrofile}' whose "
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
              ):
        if beta is None:
            beta = self.beta
        for option, sol in self.wavsols.items():
            if not len(sol.arcs):
                io_logger.warning(f"There is no arc information for option "
                                  f"'{_print_option(option, self.group_by)}', "
                                  f"keeping original fit")
                continue

            sol.refit_lines(beta=beta)
            sol.fit_function()

        return self

    def __repr__(self):
        return (f"<{super().__repr__()} Wavelength Solution. {len(self.wavsols)}x sets of "
                f"{tuple(self.group_by)}: {",".join([str(x) for x in self.wavsols.keys()])}>")

    def __call__(self,
                 data,
                 meta=None,
                 ):
        data, meta = super().__call__(data, meta=meta)

        group_key = tuple([meta[val] for val in self.group_by])
        data['wav'] = self.wavsols[group_key](data['pix'])

        return data, meta

    def plot_width(self, ncol=2,
                   axs=None):
        n_res = len(self.wavsols)
        if axs is None:
            f, axs = plt.subplots(nrows=int(np.ceil(n_res/ncol)), ncols=ncol,)

        for (option, wavsol), ax in zip(self.wavsols.items(), axs.flatten()):
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

        n_res = len(self.wavsols)
        if axs is None:
            f, axs = plt.subplots(nrows=int(np.ceil(n_res/ncol)), ncols=ncol,)

        for (option, wavsol), ax in zip(self.wavsols.items(), axs.flatten()):

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
        axs = prepare_mosaic_axes(len(labels) + 2, ncols)

        for option, sol in self.wavsols.items():
            if not (len(sol.arcs) + len(sol.fits)):
                io_logger.warning(f"No arc nor fits information to plot for "
                                  f"{_print_option(option, self.group_by)}")
                continue

            idxs = []
            for lab in sol.pixwav['label']:
                idxs.append(labels.index(lab))
            idxs.append(len(axs)-1)

            inax = np.array(axs)[np.array(idxs)]
            sol.plot_fit(axs=inax, legend_title=f"{{}} for {option}")

        axs[ncols//2].set_title(title)

    def save_in(self, directory):
        for wavsol in self.wavsols.values():
            wavsol.write(directory=directory)


def _print_option(option, grouping):
    return f"{','.join(grouping)}={','.join([str(v) for v in option])}"
