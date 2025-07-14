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
        return self.axis_collapse_with("median",
                                       axis=axis, label=label)

    def mean(self,
             axis: str | int | None = None,
             label: str | None = None,
             ):
        return self.axis_collapse_with("mean",
                                       axis=axis, label=label)

    def std(self,
            axis: str | int | None = None,
            label: str | None = None,
            ):
        return self.axis_collapse_with("std",
                                       axis=axis, label=label)

    def min(self,
            axis: str | int | None = None,
            label: str | None = None,
            ):
        return self.axis_collapse_with("min",
                                       axis=axis, label=label)

    def max(self,
            axis: str | int | None = None,
            label: str | None = None,
            ):
        return self.axis_collapse_with("max",
                                       axis=axis, label=label)

    def axis_collapse_with(self,
                           statistic: str | list[str],
                           axis: str | int | None = None,
                           label: str | None = None,
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

            ret_ad = self.axis_collapse_with(statistic[0],
                                             axis,
                                             label=label,
                                             ).axis_add_new("C",
                                                       value=val)

            for stat in statistic[1:]:
                val = f"{statistics[stat][0]} along {along_name}"
                ret_ad = ret_ad.concatenate_along("C",
                                                  self.axis_collapse_with(stat, axis=axis,
                                                                          label=label),
                                                  value=val,
                                                  )

            return ret_ad

        raise ValueError(f"Statistic ({statistic}) must be a valid string or list of strings")

    def axis_add_new(self,
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

    def axis_reorder(self,
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

    def filter(self,
               **kwargs) -> "AstroData":
        """
        Compares if the expected value of the given header item matches
        the value stored in this file header or in an axis value.

        Filtering can be customized by including one of the following options
        after the item name, separated by an underscore.

        Keyword syntax (always case insensitive):

           [AXIS_]<FIELD_NAME>[_<OPERATIOM1>[_<OPERATION2>[...]]]

           Where FIELD_NAME is the axis name if preceded with `AXIS_`; otherwise, it refers to a key in meta.
           operators can be a non-contradictory mix of

           - Strings:
            + BEGIN:     True if value starts with the given string
            + END:       True if value ends with the given string
            + ICASE:     Case unsensitive match
            + MATCH:     Case sensitive match

           - Numeric values:
            + LT / LE:   True if current value is lower (or equal) than the given number
            + GT / GE:   True if current value is greater (or equal) than the given number
            + EQUAL:     True if both values are the same

           - Other:
            + NOT:       Logical Not

        It is possible to include multiple options, this statements count as
        a logical "or" statements.

        Parameters
        ----------
        kwargs : Keyword Arguments or unpacked dictionary
            item_name_option : value

        Returns
        -------
        AstroData

        Notes
        -----
        If a header item has a "-" character on its name, use two underscores
        to represent it.

        Examples
        --------
        NAME_BEGIN_ICASE_NOT = "WASP"   (False if string starts with wasp)
        NAXIS1_GT_EQUAL = 20                  (True if NAXIS1 >= 20)
        AXIS_X_LT = 20   (True for all values of x axis less than 20)

        """
        ret = []

        axes_mask = {}

        for filter_keyword, request in kwargs.items():
            functions = []
            axis_filter = False

            # By default, it is not comparing match, but rather equality
            match = False
            equal = True
            negate = False

            lower_than = greater_than = False

            filter_keyword = filter_keyword.replace('__', '-')

            if '_' in filter_keyword:
                tmp = filter_keyword.split('_')
                filter_keyword = tmp.pop(0).lower()
                functions.extend(tmp)

            try:
                filter_keyword = filter_keyword[0].upper() + filter_keyword[1:]
                axis_filter = self.axes.index(filter_keyword) >= 0
            except IndexError:
                axis_filter = False
                raise NotImplementedError("meta indexing needs work")

            if axis_filter:
                values = self.axes[filter_keyword].values()
            else:
                try:
                    values = self[filter_keyword]
                except IndexError:
                    # Treat specially the not-found and list as filter_keyword
                    ret.append(False)
                    continue

            if isinstance(request, dict):
                raise TypeError("Filter string cannot be list (to provide multiple Ok values) or scalar only. ")
            elif isinstance(request, (tuple, list)):
                pass
            else:
                request = [request]

            functions = [f.lower() for f in functions]
            # check that either numeric or string keywords are given, not both
            if (any([f in ["match", "begin", "end"] for f in functions]) and
                    any([f in ["lt", "le", "gt", "ge"] for f in functions])):
                raise ValueError(f"cannot request both numeric and string functions in filter{functions}")

            for f in functions:
                if f[:5] == 'begin':
                    values = values[:len(request[0])]
                elif f[:3] == 'end':
                    values = values[-len(request[0]):]
                elif f[:5] == 'icase':
                    values = values.lower()
                    request = [a.lower() for a in request]
                elif f[:3] == 'not':
                    negate = True
                elif f[:5] == 'match':
                    match = True
                    equal =  False
                elif f[:5] == 'equal':
                    match = False
                    equal = True
                elif f[:3] == 'lt':
                    lower_than = True
                    equal = False
                elif f[:3] == 'gt':
                    greater_than = True
                    equal = False
                elif f[:3] == 'le':
                    lower_than = True
                    equal = True
                elif f[:3] == 'ge':
                    greater_than = True
                    equal = True
                else:
                    pa_logger.warning(f"Function '{f}' not recognized in "
                                      f"filtering, ignoring")


            # if axes filtering return a new cropped astrodata
            if not axis_filter:
                raise NotImplementedError("meta filtering needs work still")

            values_expanded = values[None, :] * np.ones([len(request), 1])
            request_expanded = np.array(request)[:, None] * np.ones([1,len(values)])
            mask = np.zeros(len(values), dtype=bool)

            if greater_than:
                mask += (values_expanded > request_expanded).sum(0, dtype=bool) != negate

            if lower_than:
                mask += (values_expanded < request_expanded).sum(0, dtype=bool) != negate

            if match:
                mask += (np.char.find(values_expanded,
                                     request_expanded)>0).sum(0, dtype=bool) != negate

            if equal:
                mask += (values_expanded == request_expanded).sum(0, dtype=bool) != negate

            if filter_keyword in axes_mask:
                axes_mask[filter_keyword] += mask
            else:
                axes_mask[filter_keyword] = mask

        ret_astrodata = self
        for filter_keyword, mask in axes_mask.items():
            axes = ret_astrodata.axes.copy()
            data = ret_astrodata.data
            axis_idx = axes.index(filter_keyword)
            original_axis = self._axes[filter_keyword]
            axes[filter_keyword].update_values(original_axis.values()[mask])
            mask_idx = np.arange(len(original_axis))[mask]

            slices = tuple([slice(None, None, None)] * axis_idx + [mask_idx]
                           + [slice(None, None, None)] * (len(axes) - axis_idx - 1))
            data = data[slices]

            ret_astrodata = AstroData(data=data, axes=axes, meta=self.meta)

        return ret_astrodata

        #     if greater_than:
        #         ret.append((True in [r < values
        #                              for r in request]) != negate)
        #     if lower_than:
        #         ret.append((True in [r > values
        #                              for r in request]) != negate)
        #     if match:
        #         ret.append((True in [r in values
        #                              for r in request]) != negate)
        #     if equal:
        #         ret.append((True in [r == values
        #                              for r in request]) != negate)
        #
        # # Returns whether the filter existed (or not, if _not function)
        # return True in ret

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
            self.axis_add_new(axis)
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

        astro_data = self.axis_reorder(transposed_dims)
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
            meta = CaseInsensitiveMeta({})

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

    new_ad = ad.axis_collapse_with(["min", "max", "med"], 'X')
    new_ad.plot('Y', label='C')

    ad.filter(x_gt=50).plot('X')
