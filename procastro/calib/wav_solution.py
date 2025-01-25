import numpy as np
from matplotlib import pyplot as plt

from procastro.misc import functions
from procastro.parents.calib import CalibBase
from procastro.logging import io_logger
import procastro as pa

__all__ = ['WavSol']

from procastro.misc.functions import use_function


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


class WavSol (CalibBase):
    def __init__(self,
                 pixwav: "pa.AstroDir | pa.AstroFile",
                 arcs=None,
                 refit=True,
                 n_sigma=2,
                 group_by='trace',
                 pixwav_function="poly:d4",
                 **kwargs):
        super().__init__(**kwargs)

        # if group_by is None:
        #     group_by = [[]]
        if not isinstance(group_by, list):
            group_by = [group_by]
        self.group_by = group_by

        if isinstance(pixwav, pa.AstroFile):
            pixwav = pa.AstroDir([pixwav])
        if not pixwav[0].spectral:
            raise TypeError("pixwav must have spectral data")

        self.function = {}
        self.pixwavs = {}
        self.mask = {}
        for astrofile in pixwav.mosaic_by(*group_by, in_place=False):
            option = tuple(astrofile.values(*group_by, single_in_list=True))
            table = astrofile.data
            pix = table['pix']
            if option in self.function:
                io_logger.warning(f"More than one of tables in pixwav ({pixwav}) has identical grouping")
            pre_fnc = functions.use_function(pixwav_function, pix, table['wav'])

            if n_sigma > 0:
                residuals = table['wav'] - pre_fnc(pix)
                std = residuals.data.std()
                mask = np.abs(residuals) < n_sigma*std
                fcn = functions.use_function(pixwav_function, pix[mask], table['wav'][mask])
            else:
                fcn = pre_fnc
                mask = table['wav'].data * 0 == 0

            self.mask[option] = mask
            self.function[option] = fcn
            self.pixwavs[option] = astrofile

        self.function_name = pixwav_function

    def __repr__(self):
        return (f"<{super().__repr__()} Wavelength Solution. {len(self.function)}x sets of "
                f"{self.group_by}: {list(self.function.keys())}>")

    def __call__(self,
                 data,
                 meta=None,
                 ):
        data, meta = super().__call__(data, meta=meta)

        group_key = tuple([meta[val] for val in self.group_by])
        data['wav'] = self.function[group_key](data['pix'])

        return data, meta

    def residuals(self,
                  reference='lin_reg',
                  alternate_functions=None,
                  plot_kw=None,
                  plot_other=None,
                  plot_fcn=None,
                  axs=None,
                  ):
        """

        Parameters
        ----------
        alternate_functions
        reference
        plot_other
        plot_fcn
        plot_kw:
           'extra_percent' at both sides of extremes when plotting
           'ncol' number of columns for mosaic of axes
        axs
        """
        plot_kw = plot_kw or {}
        plot_fcn = plot_fcn or {}
        plot_other = plot_other or {}

        plot_kw = {'extra_percent': 5,
                   'ncol': 2,
                   'ls': '',
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

        extra_percent = plot_kw.pop('extra_percent')
        ncol = plot_kw.pop('ncol')

        if alternate_functions is None:
            alternate_functions = ['poly:d2',
                                   'poly:d3',
                                   f'spline:s{int(0.8*len(list(self.pixwavs.values())[0]))}'
                                   ]

        n_res = len(self.function)
        if axs is None:
            f, axs = plt.subplots(nrows=int(np.ceil(n_res/ncol)), ncols=ncol,)

        for (option, astrofile), ax in zip(self.pixwavs.items(), axs.flatten()):
            data = astrofile.data
            pix = data['pix']

            mxpix = max(pix)
            mnpix = min(pix)

            ref = use_function(reference, pix, data['wav'])
            fcn = self.function[option]
            mask = self.mask[option]
            x = np.linspace(mnpix, mxpix, 40)

            ax.plot(pix[mask], (data['wav'] - ref(pix))[mask],
                    zorder=10,
                    label="Included lines",
                    **plot_kw)
            if mask.sum() != len(mask):
                ax.plot(pix[~mask], (data['wav'] - ref(pix))[~mask],
                        zorder=10,
                        label="Excluded lines",
                        **plot_kw | {'marker': 'X'})
            ax.plot(x, fcn(x) - ref(x),
                    zorder=9,
                    label=f"{self.function_name} selected",
                    **plot_fcn)

            for linestyle, other_fcn in zip(linestyle_tuple, alternate_functions):
                ax.plot(x, use_function(other_fcn, data['pix'], data['wav'])(x) - ref(x),
                        label=other_fcn,
                        zorder=8,
                        **plot_other | {'ls': linestyle[1]},
                        )
            ax.legend(fontsize=8, title=f"{','.join(self.group_by)}: {','.join([str(op) for op in option])}", title_fontsize=10)

        axs[0][0].set_title(f"Wavelength residuals with respect to {ref.short()} fit")
        for ax in axs[-1]:
            ax.set_xlabel("pixel")
