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
                 label: str = None,
                 ):
        if data is not None:
            self._data = data
            if meta is None:
                raise ValueError("Meta needs to be specified if data is given")
        if meta is not None:
            self._meta = data

        self._axes: AstroAxes | None = axes
        self._label = label

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

    def __getitem__(self, item):
        return self.data[item]

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
                         label=self._label)

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
                         label=self._label)

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

        y_ax = self._axes[0]
        x_ax = self._axes[1]
        im = ax.imshow(self.data, cmap=colormap, vmin=vmin, vmax=vmax,
                  extent=x_ax.lims() + y_ax.lims())
        ax.set_xlabel(x_ax.label())
        ax.set_ylabel(y_ax.label())

        if colorbar is not None:
            colorbar['label'] = self._label
            ax.figure.colorbar(im, **colorbar)

        if title is not None:
            ax.set_title(title)

        if show:
            ax.figure.show()

    def plot(self,
             x_var: str = None,
             y_var: str = "N",
             ax: Axes = None,
             title: str = None,
             show=True,
             ):

        if ax is None:
            f, ax = plt.subplots()

        if x_var is None:
            if self.ndims == 1:
                x_var = self._axes[0].acronym
            else:
                raise ValueError(f"Need to specify x-axis among: {self._axes.str_available()}")

        if y_var == 'N' or y_var == 'F':
            x_axis_idx = self._axes.index(x_var)
            transposed_dims = [x_axis_idx] + [i for i in range(self.ndims) if i!=x_axis_idx]
            y_axis = DataAxis(self.data.transpose(transposed_dims), label=self._label)
            x_axis = self._axes[x_axis_idx]

        elif x_var == 'N' or x_var == 'F':
            y_axis_idx = self._axes.index(y_var)
            transposed_dims = [y_axis_idx] + [i for i in range(self.ndims) if i!=y_axis_idx]
            x_axis = DataAxis(self.data.transpose(transposed_dims), label=self._label)
            y_axis = self._axes[y_axis_idx]

        else:
            raise ValueError("Either x- or y-axis needs to be the data axis (identify by N or F)")

        ax.plot(x_axis.values, y_axis.values)
        ax.set_xlabel(x_axis.label())
        ax.set_ylabel(y_axis.label())

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
                   label: str = None,
                   ) -> "AstroData":
        """

        Parameters
        ----------
        data
        meta
        axes: str
           String specifying axes types

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

        return cls(data=data, meta=meta, axes=axes_list, label=label)


if __name__ == "__main__":

    arr = np.arange(200)[None, :] * np.ones(400)[:, None]

    ad = AstroData.from_array(arr, label='Flux')
    print(ad[10:40, 50:60])

    ad.plot('Y', 'N',
            #filter={'Y': 'median'},
            )
    ad.imshow(colorbar={'orientation': 'horizontal'})

    print(ad.median('X'))
    ad.median('X').plot()
