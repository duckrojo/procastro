import re
from itertools import repeat
from typing import Any

import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from procastro.axis.axes import AstroAxes
from procastro.axis.data import DataAxis
from procastro.config import pa_logger
from procastro.data.utils import DictInitials, CaseInsensitiveMeta


class AstroData:
    unit = None

    def __str__(self):
        return f'AstroData cube{ f'({self.unit})' if self.unit is not None else ''}. {self._axes}'

    def __init__(self,
                 data=None,
                 meta: CaseInsensitiveMeta=None,
                 axes: AstroAxes | None = None,
                 ):
        if data is not None:
            self._data = data
            if meta is None or axes is None:
                raise ValueError("Meta and Axes needs to be specified if data is given")

        self._meta = meta

        if axes.shape != data.shape:
            raise ValueError(f"Axes shape ({axes.shape}) and data shape ({data.shape}) do not match")
        self._axes = axes

    @property
    def ndims(self):
        return len(self._axes)

    @property
    def shape(self):
        return self.data.shape

    @property
    def meta(self):
        """Returns a copy of the meta information"""
        return CaseInsensitiveMeta(self._meta)

    @property
    def data(self):
        """Returns a copy of the data"""
        return np.array(self._data)

    @property
    def axes(self):
        """Returns a copy of the axes"""
        return AstroAxes(self._axes)

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
               label: str | None = None,
               ):
        return self.axis_collapse("median",
                                  axis=axis, label=label)

    def mean(self,
             axis: str | int | None = None,
             label: str | None = None,
             ):
        return self.axis_collapse("mean",
                                  axis=axis, label=label)

    def std(self,
            axis: str | int | None = None,
            label: str | None = None,
            ):
        return self.axis_collapse("std",
                                  axis=axis, label=label)

    def min(self,
            axis: str | int | None = None,
            label: str | None = None,
            ):
        return self.axis_collapse("min",
                                  axis=axis, label=label)

    def max(self,
            axis: str | int | None = None,
            label: str | None = None,
            ):
        return self.axis_collapse("max",
                                  axis=axis, label=label)

    def axis_collapse(self,
                      statistic: str | list[str],
                      axis: str | int | None = None,
                      label: str | None = None,
                      mask: np.ndarray = None,
                      ):
        """
        Collapse requested axis using a median

        Parameters
        ----------
        statistic : str, list[str]
          Statistic to be used for the collapse. Available options: median, std, min, max
        label: str
          New name of data value
        axis: str, int
          Axis along which the median is computed.

        Returns
        -------
        an AstroData instance with one less axis.

        """
        statistics = DictInitials(median = ("Median", np.median),
                                  mean = ("Mean", np.mean),
                                  min = ("Minimum", np.min),
                                  std = ("Standard deviation", np.std),
                                  max = ("Maximum", np.max),
                                  )

        if mask is not None:
            if len(self.axes[axis]) != len(mask):
                raise ValueError("Mask must have same length as chosen "
                                 f"axis {self.axes[axis].acronym}"
                                 f" ({len(mask)} vs {len(self.axes[axis])})")
            raise NotImplementedError("Mask not implemented yet")
        else:
            mask = np.ones(len(self.axes[axis])) == 1


        if isinstance(statistic, str):
            name, function = statistics[statistic]

            if axis is None:
                return function(self.data)

            axis = self._axes.index(axis)

            data_collapsed = function(self.data, axis=axis)
            axes_collapsed = self._axes.removed(axis)
            meta = self.meta
            if label is None:
                label = f"{name} along {self._axes[axis].short()}"
            meta['datatype'] = label

            return AstroData(data=data_collapsed, meta=meta,
                             axes=axes_collapsed,
                             )

        if isinstance(statistic, list):
            along_name = self.axes[axis].short()
            val = f"{statistics[statistic[0]][0]} along {along_name}"
            if label is None:
                label = ""

            ret_ad = self.axis_collapse(statistic[0],
                                        axis,
                                        label=label,
                                        ).with_new_axis("C",
                                                        value=val)

            for stat in statistic[1:]:
                val = f"{statistics[stat][0]} along {along_name}"
                ret_ad = ret_ad.concatenate_along("C",
                                                  self.axis_collapse(stat, axis=axis,
                                                                     label=label),
                                                  value=val,
                                                  )

            return ret_ad

        raise ValueError(f"Statistic ({statistic}) must be a valid string or list of strings")

    def concatenate_along(self,
                          axis,
                          astrodata: "AstroData",
                          value: str | None = None,
                          ) -> "AstroData":
        """

        Parameters
        ----------
        axis
        astrodata
        value

        Returns
        -------

        """
        try:
            along = self.axes.index(axis)
        except IndexError:
            self.with_new_axis(axis)
            along = self.axes.index(axis)

        required_shape = list(self.shape)
        required_shape.pop(along)

        if astrodata.shape != tuple(required_shape):
            raise ValueError(f"astrodata <{astrodata}> must have shape {tuple(required_shape)},"
                             f" not {astrodata.shape}")

        new_shape = astrodata.shape[:along] + (1,) + astrodata.shape[along:]
        data = np.concatenate((self.data, astrodata.data.reshape(new_shape)), axis=along)

        axes = self.axes
        axes[axis].append(value, in_place=True)

        return AstroData(data=data, meta=self.meta, axes=axes)

    def with_new_axis(self,
                      axis_type,
                      value: Any = 0,
                      ):
        data = self.data
        new_shape = data.shape + (1,)

        new_data = data.reshape(new_shape)
        new_axes = self.axes.add_axis(axis_type, value=value)

        return AstroData(data=new_data,
                         axes=new_axes,
                         meta=self.meta,
                         )

    def reorder(self,
                axes: str | list[int],
                ):
        """
        Reorder the axes according to list of integers, or named string with acronyms (e.g. 'XYTSw')

        Parameters
        ----------
        axes: str, list[int]
          either a list of integers identifying each of the axes by their current position, or a string with acronyms

        Returns
        -------

        """
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

            if len(iterators[0]) > 20:
                pa_logger.info(f"Warning: a high number of curves ({len(iterators[0])})"
                              f" were requested with individual label, plots will be slow")
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

    new_ad = ad.axis_collapse(["min", "max", "med"], 'X')
    new_ad.plot('Y', label='C')
