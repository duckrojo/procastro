import re
from itertools import repeat

import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from procastro.axis.axes import AstroAxes
from procastro.axis.data import DataAxis


class AstroData:
    unit = None

    def __str__(self):
        return f'AstroData cube{ f'({self.unit})' if self.unit is not None else ''}. {self._axes}'

    def __init__(self,
                 data=None,
                 meta=None,
                 axes: AstroAxes|None = None,
                 ):
        if data is not None:
            self._data = data
            if meta is None or axes is None:
                raise ValueError("Meta and Axes needs to be specified if data is given")

        self._meta = meta
        self._axes = axes

    @property
    def ndims(self):
        return len(self._axes)

    @property
    def meta(self):
        return self._meta

    @property
    def data(self):
        return self._data

    @property
    def axes(self):
        return self._axes

    def __contains__(self, item):
        return item in self._meta

    def __getitem__(self, item):
        if isinstance(item, (int, slice, tuple)):
            return self.data[item]
        elif isinstance(item, str):
            return self.meta[item]
        else:
            raise TypeError(f"item '{item}' of invalid type")

    ######################################
    #
    # reshaping data array

    def median(self,
               axis: str | int | None = None,
               ):
        if axis is None:
            return np.median(self.data)

        axis = self._axes.index(axis)

        data_med = np.median(self.data, axis=axis)
        axes_med = self._axes.removed(axis)

        return AstroData(data=data_med, meta=self._meta,
                         axes=axes_med,
                         )

    def std(self,
            axis: str | int | None = None,
            ):
        if axis is None:
            return np.std(self.data)

        axis = self._axes.index(axis)

        data_med = np.std(self.data, axis=axis)
        axes_med = self._axes.removed(axis)

        return AstroData(data=data_med, meta=self._meta,
                         axes=axes_med,
                         )

    def reorder(self,
                axes: str | list[int],
                ):
        new_axes = []

        if isinstance(axes, str):
            for axis in re.findall("([A-Z][a-z]?)", axes):
                new_axes.append(self.axes.index(axis))
        elif isinstance(axes, (list, tuple)):
            for axis in axes:
                new_axes.append(self.axes.index(axis))

        if len(new_axes) != self.ndims:
            raise TypeError(f"Axis '{axes}' must contain as many dimensions as axes")

        return AstroData(data=self.data.transpose(new_axes), meta=self.meta,
                         axes=AstroAxes([self.axes[i] for i in new_axes],),
                         )

    ######################################
    #
    # Plotting routines

    def imshow(self,
               colormap=None,
               zscale=True,
               ax=None,
               show=True,
               title=None,
               colorbar: dict | None = None,
               ):
        """Shows an image using a colormap. Cube must have been compacted to 2D before (2 AstroAxis)"""

        if len(self._axes) != 2:
            raise ValueError("AstroData must have only 2 axes before making an image")

        vmin = vmax = None
        if zscale:
            vmin, vmax = ZScaleInterval().get_limits(self.data)

        if ax is None:
            f, ax = plt.subplots()

        data = self.data
        y_ax = self._axes[0]
        x_ax = self._axes[1]

        # force natural orientation for XY-identified axis
        if y_ax.acronym == "X" and x_ax.acronym == "Y":
            data = data.transpose()
            y_ax, x_ax = x_ax, y_ax

        im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax,
                  extent=x_ax.lims() + y_ax.lims())
        ax.set_xlabel(x_ax.label())
        ax.set_ylabel(y_ax.label())

        if colorbar is not None:
            if 'datatype' in self._meta:
                colorbar['label'] = self._meta['datatype']
            ax.figure.colorbar(im, **colorbar)

        if title is not None:
            ax.set_title(title)

        if show:
            ax.figure.show()

    def plot(self,
             x_var: str | int = None,
             y_var: str | int = "N",
             ax: Axes = None,
             title: str = None,
             label: str = None,
             show=True,
             ):

        if ax is None:
            f, ax = plt.subplots()

        if x_var is None:
            if self.ndims == 1:
                x_var = self._axes[0].acronym
            else:
                raise ValueError(f"Need to specify x-axis among: {self._axes.str_available()}")

        data_on_vertical = y_var == 'N' or y_var == 'F'
        if not data_on_vertical and (x_var != 'N' and y_var != 'F'):
            raise ValueError("Either x- or y-axis needs to be the data axis (identify by N or F)")

        other_idx = self._axes.index(x_var) if data_on_vertical else self._axes.index(y_var)
        label_idx = None if label is None else self._axes.index(label)

        # choose the new order of dimensions, starting with the axis against which to plot,
        # then the axis with the label
        do_per_label = False
        transposed_dims = []
        if label_idx is not None:
            do_per_label = True
            transposed_dims.append(label_idx)
        transposed_dims.append(other_idx)

        transposed_dims += [i for i in range(self.ndims) if i not in transposed_dims]

        astro_data = self.reorder(transposed_dims)
        datatype = astro_data['datatype'] if 'datatype' in astro_data else ''
        data_axis = DataAxis(astro_data.data, label=datatype)
        other_axis = astro_data.axes[1 if do_per_label else 0]

        if label_idx is not None:
            if data_on_vertical:
                iterators = (astro_data.axes[0].values(), repeat(astro_data.axes[1].values()), astro_data.data)
            else:
                iterators = (astro_data.axes[0].values(), astro_data.data, repeat(astro_data.axes[1].values()))

            for val, xx, yy in zip (*iterators):
                ax.plot(xx, yy, label=val)
            ax.legend(title=astro_data.axes[0].label())
        else:
            if data_on_vertical:
                ax.plot(other_axis.values(), data_axis.values())
            else:
                ax.plot(data_axis.values(), other_axis.values())

        if data_on_vertical:
            ax.set_xlabel(other_axis.label())
            ax.set_ylabel(data_axis.label())
        else:
            ax.set_xlabel(data_axis.label())
            ax.set_ylabel(other_axis.label())

        if title is not None:
            ax.set_title(title)

        if show:
            ax.figure.show()

    ########################################################
    #
    # initialization methods

    @classmethod
    def from_array(cls,
                   data: np.ndarray,
                   meta: dict = None,
                   axes: str = None,
                   data_type: str | None = None,
                   ) -> "AstroData":
        """

        Parameters
        ----------
        data
        meta
        axes: str
           String specifying axes types
        data_type: str
           Data type. Typically: Flux, Counts, ADU, DN

        Returns
        -------

        """
        if axes is None:
            match len(data.shape):
                case 1:
                    axes = "X"
                case 2:
                    axes = "XY"
                case _:
                    raise ValueError("Axes type needs to be specified explicitly if more than 2 axes")

        axes_list = AstroAxes.from_linear(axes, data.shape)
        if len(axes_list) != len(data.shape):
            raise ValueError(f"As many axes ({len(axes_list)}) as data dimensions "
                             f"must be specified ({len(data.shape)})")

        if meta is None:
            meta = {}

        if data_type is not None:
            meta["datatype"] = data_type

        return cls(data=data, meta=meta, axes=axes_list)


if __name__ == "__main__":

    arr = np.arange(200)[None, :] * np.ones(400)[:, None]

    ad = AstroData.from_array(arr, data_type='Flux')
    print(ad[10:40, 50:60])

    ad.plot('Y', 'N',
            #filter={'Y': 'median'},
            )
    ad.imshow(colorbar={'orientation': 'horizontal'})

    print(ad.median('X'))
    ad.median('X').plot()
    ad.plot('Y', label='X')
