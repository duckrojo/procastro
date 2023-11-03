# (c) 2023 Patricio Rojo

from pathlib import Path
from typing import Union

import inspect
import matplotlib
from matplotlib import patches, transforms
from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent
from matplotlib.axes import Axes
import numpy as np
import procastro as pa
from functools import wraps as _wraps
import re

__all__ = ['BindingsFunctions']

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml
import tomli_w
from typing import Optional, Callable, Any

matplotlib.rcParams['keymap.xscale'].remove('L')
matplotlib.rcParams['keymap.quit'].remove('q')
matplotlib.rcParams['keymap.save'].remove('s')

TwoValues = tuple[float, float]
FunctionArgs = tuple[Callable, list[Any]]
FunctionArgsKw = tuple[Callable, list[Any], dict]


def _clear_axes(*axes_right):
    """Clear examination area for a specified"""
    for ax, right in axes_right:
        ax.cla()
        ax.set_aspect("auto")
        if right:
            ax.yaxis.tick_right()


def _csv_str_to_tuple(value):
    value = re.sub(' *, *', ',', value)
    if ',' in value:
        return value.split(',')

    return value,


def _allow_keep_val(params):
    """requires lists of parameters that are allowed to remain between calls."""

    def wrapper1(method):
        # noinspection PyUnusedLocal
        def do_nothing(*args, **kwargs):
            return

        @_wraps(method)
        def wrapper(instance, xy, **kwargs):
            name = method.__name__
            pre_run = name in instance.last_dict

            if 'xy' in params:
                kwargs['xy'] = xy
            for prm in params:
                if prm in kwargs:
                    if kwargs[prm] is None:
                        # do not enter the function if it has not been called previously
                        # and some previous values are requested
                        if not pre_run:
                            return do_nothing
                        else:
                            kwargs[prm] = instance.last_dict[name][prm]
                else:
                    kwargs[prm] = inspect.signature(method).parameters[prm].default

            to_store_dict = {k: v for k, v in kwargs.items() if k in params}
            xy = kwargs.pop('xy', xy)

            ret = method(instance, xy, **kwargs)
            instance.last_dict[name] = to_store_dict

            return ret
        return wrapper
    return wrapper1


def _pre_process(method):
    """Apply pre-process to data, this allows general treatment of 3D, 2D functions and others"""
    # pre_process: Union[None,
    #                    Callable, tuple[Callable],
    #                    FunctionArgs, FunctionArgsKw] = None,  # This is fully processed
    #                                                           # by the decorator
    @_wraps(method)
    def wrapper(instance, data, *args, **kwargs):
        pre_process = kwargs.pop('pre_process')
        if pre_process is None:
            pass
        elif isinstance(pre_process, tuple):
            pp_args = []
            pp_kwargs = {}
            if len(pre_process) > 1:
                pp_args = pre_process[1]
                if len(pre_process) == 3:
                    pp_kwargs = pre_process[2]
                elif len(pre_process) > 3:
                    raise TypeError(f"pre_process keyword tuple ({pre_process}) has too many components"
                                    f"Expected: (Callable), (Callable, List), or (Callable, List, Dict)")
            data = pre_process[0](data, *pp_args, **pp_kwargs)
        else:
            raise TypeError(f"pre_process keyword type {pre_process} is not correct. "
                            f"Expected: None, Callable, (Callable), (Callable, List), or (Callable, List, Dict)")
        return method(instance, data, *args, **kwargs)

    return wrapper


class BindingsFunctions:
    def __init__(self,
                 axes_data,
                 axes_exam,
                 config_file='interactive.toml',
                 cb_data=None,
                 cb_exam=None,
                 title: str = '',
                 ):
        self._last_scale_exam = None
        self._last_scale_data = None

        self.last_dict = {}

        self._config_file = config_file
        self._key_options = None
        self._data_2d = None

        self._config = {}
        self._temporal_artist = []
        self._cid = {}

        self._mark_patches = {'point': [],
                              }
        self._marks = {'point': [],
                       }

        if axes_data is None or isinstance(axes_data, int):
            f = plt.figure(axes_data, figsize=(16, 8))
            gs = f.add_gridspec(2, 2, height_ratios=(12, 1),
                                left=0.05, right=0.95, top=0.95, bottom=0.05,
                                wspace=0.05, hspace=0.15)
            axes_data = f.add_subplot(gs[0, 0])
            axes_exam = f.add_subplot(gs[0, 1])

            cb_data = f.add_subplot(gs[1, 0])
            cb_exam = f.add_subplot(gs[1, 1])

            f.patch.set_facecolor("navajowhite")
            axes_data.set_title(title)
            f.show()

        elif not isinstance(axes_data, Axes):
            raise NotImplementedError(f"Axes type '{axes_data}' not supported")

        self._axes_2d = axes_data
        self._axes_exam = axes_exam
        self._colorbar_data = cb_data
        self._colorbar_exam = cb_exam

        self._extra_transform_data = transforms.Affine2D()

        self._options_history = []

    def set_data_2d(self,
                    data: Optional[np.ndarray],
                    imshow_kwargs: Optional[dict] = None,
                    scale: Optional[str] = None,
                    ):
        """Set data content for internal treatment of key bindings.
        Due to variety of contrasts This function does not plot the data, only its colorbar if such axes exists.

        Parameters
        ----------
        scale:
           passed to self.change_scale_data
        data:
           NumPy array with data
        imshow_kwargs: dict, optional
           If not None, then call imshow with the given dictionary as keyword arguments. If None, no imshow() is called.

        """

        if data is not None:
            self._data_2d = data

        if scale is not None:
            self.change_scale_data(None, scale=scale)

        if imshow_kwargs is not None:
            self.clear_data()
            self._axes_2d.imshow(self._data_2d, **imshow_kwargs)
            self._axes_2d.figure.canvas.draw_idle()

        if self._colorbar_data is not None:
            fig = self._axes_2d.figure
            fig.colorbar(self._axes_2d.get_images()[0],
                         cax=self._colorbar_data,
                         orientation="horizontal",
                         )
            self._colorbar_data.figure.canvas.draw_idle()

    def _pix_from_percent(self, param, prop=''):
        """Computes radius in pixel given percent of image (*percent) if the *pixels property is not set"""
        if f'{prop}pixels' not in param or param[f'{prop}pixels'] == 0:
            yy, xx = self._data_2d.shape
            return int(param[f'{prop}percent'] * max(xx, yy) / 200)
        return param[f'{prop}pixels']

    def _mark_temporal_in_2d(self,
                             props,
                             mark=None,
                             box=None,
                             ):

        tr = self._axes_2d.get_images()[0].get_transform()
        if self._axes_2d is None:
            return
        self.clear_temp()

        if mark is not None:
            self._temporal_artist.extend(self._axes_2d.plot(*mark, props['marker'],
                                                            color=props['color'],
                                                            alpha=props['alpha'],
                                                            transform=tr,
                                                            )
                                         )
        if box is not None:
            rect = patches.Rectangle((box[0], box[2]), width=box[1]-box[0], height=box[3]-box[2],
                                     color=props['color'],
                                     alpha=props['alpha'],
                                     transform=tr,
                                     )
            self._axes_2d.add_patch(rect)
            self._temporal_artist.append(rect)

        self._axes_2d.figure.canvas.draw_idle()

    def _load_config(self,
                     reset: bool = False):

        def update_vals(new_config, old_config):
            for key, item in new_config.items():
                if isinstance(item, dict):
                    item = update_vals(item, old_config[key])
                old_config[key] = item
            return old_config

        try:
            self._config = toml.loads(Path(pa.defaults_confdir(self._config_file)).read_text(encoding='utf-8'))
        except IOError:
            return False
        if reset:
            print("Forced reset: loaded factory defaults")
        else:
            file = pa.user_confdir(self._config_file)
            try:
                new = toml.loads(Path(file).read_text(encoding='utf-8'))
                self._config = update_vals(new, self._config)
                print(f"Loaded configuration from: {file}")
            except toml.TOMLDecodeError:
                print(f"Skipping configuration from corrupt local config ({file}). "
                      f"It is recommended to save new version.")

        return True

    def _save_config(self,
                     ):
        file = pa.user_confdir(self._config_file)
        with open(file, 'wb') as fp:
            tomli_w.dump(self._config, fp)
        print(f"Saved configuration to: {file}")

    ############################
    # start-end interactive mode
    #
    ############################

    def disconnect(self,
                   verbose=True,
                   close_plot: bool = True):
        if verbose:
            print("... exiting interactive mode")
        self._axes_2d.figure.canvas.stop_event_loop()

        for cid in self._cid.values():
            self._axes_2d.figure.canvas.mpl_disconnect(cid)

        if close_plot:
            plt.close(self._axes_2d.figure.number)
            if self._axes_exam:
                plt.close(self._axes_exam.figure.number)

    def connect(self, verbose=True):
        self._cid['key_press'] = self._axes_2d.figure.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._axes_2d.figure.canvas.draw_idle()
        if self.axes_exam is not None:
            self._axes_exam.figure.canvas.draw_idle()

        if verbose:
            print("Entering interactive mode ('?' for help)")
        self._axes_2d.figure.canvas.start_event_loop()

    def _on_key_press(self, event):
        self._options_choose(event)

    ############################
    # Callable functions
    #
    ############################

    def set_marks(self,
                  marks: list[TwoValues],
                  mark_type: str = 'point',
                  redraw: bool = True,
                  ):
        self._marks[mark_type] = marks

        if redraw:
            self.redraw_marks(only=mark_type)

    def draw_mark(self,
                  mark_type: str,
                  pos: int = -1,
                  ):
        props = self._config['tool']['marking']
        colors = _csv_str_to_tuple(props['color'])
        # if given a negative position, then count from the end
        if pos < 0:
            pos += len(self._marks[mark_type])

        color = colors[pos] if pos < len(colors) else colors[-1]

        if mark_type == 'point':
            radius = self._pix_from_percent(props, 'point_radius_')
            patch = patches.Circle(self._marks[mark_type][pos],
                                  radius=radius,
                                  alpha=props['alpha'],
                                  color=color)
        else:
            raise NotImplementedError(f"Mark type '{mark_type}' is not implemented for now. Only 'point'")

        self._mark_patches[mark_type].append(patch)
        self._axes_2d.add_patch(patch)
        self._axes_2d.figure.canvas.draw_idle()

    def redraw_marks(self,
                     only: Optional[str] = None,
                     ):

        if only is None:
            for m in self._marks.keys():
                self.redraw_marks(only=m)
            return

        self.delete_marks_from_2d(None,
                                  clear_data=False,
                                  only=only)

        marks = self._marks[only]
        for idx in range(len(marks)):
            self.draw_mark(only, pos=idx)

    def clear_temp(self):
        """Clear temporal artists in data area"""
        if self._temporal_artist:
            for t in self._temporal_artist:
                t.remove()
            self._temporal_artist = []

    def clear_exam(self, temporal=True):
        """Clear exam area for a new plot, and its colorbar if there is any"""
        axes = [(self._axes_exam, False)]
        if self._colorbar_exam is not None:
            axes.append((self._colorbar_exam, True))
        _clear_axes(*axes)

        if temporal:
            self.clear_temp()

        self.last_dict = {k: v for k, v in self.last_dict.items() if 'exam' not in k}

    def clear_data(self, keep_title=True):
        """Clear data area for a new plot, and its colorbar if there is any"""
        axes = [(self._axes_2d, False), (self._axes_exam, True)]
        if self. _colorbar_data is not None:
            axes.append((self._colorbar_data, False))
        title = self._axes_2d.get_title()
        _clear_axes(*axes)
        if keep_title:
            self._axes_2d.set_title(title)

    @property
    def axes_data(self):
        return self._axes_2d

    @property
    def axes_exam(self):
        return self._axes_exam

    def change_scale_data(self, xy, scale='zscale'):
        """Change the scale of the data in 2D mapping"""

        scale_options = ['minmax', 'zscale']
        if scale == 'cycle':
            try:
                scale = scale_options[(scale_options.index(self._last_scale_data) + 1) % len(scale_options)]
            except ValueError:
                scale = scale_options[0]
        self._last_scale_data = scale

        if scale == 'zscale':
            vmin, vmax = pa.zscale(self._data_2d)
        elif scale == 'minmax':
            vmin, vmax = self._data_2d.min(), self._data_2d.max()
        else:
            raise ValueError(f"Unsupported scale '{scale}'")

        self.set_data_2d(None, imshow_kwargs={"vmin": vmin, "vmax": vmax})

    @_allow_keep_val(["xy", "text", "stamp_rad", "scale"])
    def zoom_exam_2d(self,
                     xy: TwoValues,
                     scale: str = 'original',
                     text: Optional[bool] = False,
                     stamp_rad: Optional[int] = 0,
                     ):
        """"returns a zoom of the stamp"""

        self.clear_exam()

        cxy = (int(xy[0]), int(xy[1]))
        props = self._config['tool']['zoom']

        if stamp_rad == 0:
            stamp_rad = self._pix_from_percent(self._config['stamp'])

        data_extents = cxy[0] - stamp_rad, cxy[0] + stamp_rad, cxy[1] - stamp_rad, cxy[1] + stamp_rad
        stamp = self._data_2d[data_extents[2]:data_extents[3], data_extents[0]:data_extents[1]]

        min_value = np.min(stamp)
        max_value = np.max(stamp)
        min_idxs = np.transpose((stamp == min_value).nonzero())
        max_idxs = np.transpose((stamp == max_value).nonzero())
        std_value = np.std(stamp)
        iqr_value = np.subtract(*np.percentile(stamp, [75, 25]))
        mean_value = np.mean(stamp)
        median_value = np.median(stamp)

        scale_options = ['zscale', 'minmax', 'original']
        if scale == 'cycle':
            scale = scale_options[(scale_options.index(self._last_scale_exam) + 1) % len(scale_options)]
        if scale == 'zscale':
            vmin, vmax = pa.zscale(self._data_2d, trim=0.02)
        elif scale == 'minmax':
            vmin, vmax = min_value, max_value
        elif scale == 'original':
            vmin, vmax = self._axes_2d.get_images()[0].get_clim()
        else:
            raise ValueError(f"zoom called with an unsupported 'scale' option: {scale}")
        self._last_scale_exam = scale
        mid = (vmin + vmax)/2

        im = self._axes_exam.imshow(self._data_2d,
                                    cmap=props['colormap'],
                                    vmin=vmin, vmax=vmax,
                                    origin='lower',
                                    )
        transform = self._extra_transform_data + im.get_transform()
        im.set_transform(transform)

        bottom_left = data_extents[0], data_extents[2]
        upper_right = data_extents[1], data_extents[3]
        extremes = self._extra_transform_data.transform([bottom_left, upper_right])
        new_extremes = np.array([extremes.min(0), extremes.max(0)])
        extents = list(new_extremes[:, 0] - 0.5) + list(new_extremes[:, 1] - 0.5)

        self._axes_exam.set_xlim(*extents[0:2])
        self._axes_exam.set_ylim(*extents[2:4])

        ny, nx = self._data_2d.shape
        shadow_area = [[extents[0], extents[2]], [extents[0], extents[3]],
                       [extents[1], extents[3]], [extents[1], extents[2]],
                       [0, extents[2]], [0, 0], [nx, 0], [nx, ny], [0, ny], [0, extents[2]],
                       ]
        polygon = patches.Polygon(shadow_area, closed=True, facecolor="black", alpha=0.5,
                                  transform=transform)
        self.axes_exam.add_patch(polygon)

        ax = self.axes_exam
        stat_mark = props['stat']
        border = stat_mark['from_border']
        if self._colorbar_exam is not None:
            ax.figure.colorbar(im, cax=self._colorbar_exam,
                               orientation='horizontal',
                               )

            for value, vert, label in zip([max_value, min_value, median_value, mean_value],
                                          [0.5, 0.5, 0.2, 0.8],
                                          ["max", "min", "median", "mean"],
                                          ):
                sym = "d"
                if value < mid:
                    color = props['facecolor_light']
                    ha = "left"
                    label = f" {value:.1f} {label}"
                else:
                    color = props['facecolor_dark']
                    ha = 'right'
                    label = f"{label} {value:.1f} "

                if value < vmin:
                    label = r" $\leftarrow$" + f"{label}"
                    value = vmin
                    sym = "|"
                elif value > vmax:
                    label = f"{label}" + r"$\rightarrow$ "
                    value = vmax
                    sym = "|"

                self._colorbar_exam.plot(value, vert, sym,
                                         color=color)
                self._colorbar_exam.text(value, vert, label,
                                         ha=ha, va='center',
                                         color=color)

        rectangle = patches.Rectangle
        color = stat_mark['color']
        for idx, label in [[min_idxs, "min"], [max_idxs, "max"]]:
            for y, x in idx:
                y += data_extents[2]-0.5
                x += data_extents[0]-0.5
                ax.add_patch(rectangle((x + border, y + border),
                                       1 - 2*border, 1 - 2*border,
                                       alpha=stat_mark['alpha'], linewidth=stat_mark['linewidth'],
                                       color=color, fill=False,
                                       transform=transform,
                                       )
                             )
                ax.text(x + 2*border, y, label, ha="left", va="bottom", color=color,
                        transform=transform,
                        )

        x0 = extents[0] + 0.5
        y0 = extents[2] + 0.5
        if text:
            y, x = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
            for xx, yy in zip(y.flatten(), x.flatten()):
                val = stamp[yy, xx]
                color = props['facecolor_light' if val < mid else 'facecolor_dark']
                ax.text(xx + x0, yy + y0, f"{val:.1f}",
                        color=color, ha='center', va='center',
                        size=props['fontsize'],
                        transform=transform,
                        )

        sep = r"$\bullet$"
        pa.set_plot_props(self._axes_exam,
                          title=f"max:{max_value} {sep} min:{min_value} {sep}"
                                f" mean:{mean_value:.1f} {sep} median:{median_value} {sep}"
                                f" std:{std_value:.1f} {sep} iqr:{iqr_value}",
                          ylabel=f"{scale[0].upper()}{scale[1:]} scale",
                          )

        self._mark_temporal_in_2d(self._config['mark'][props['mark_data']],
                                  box=data_extents,
                                  )

    def plot_radial_exam_2d(self,
                            xy: TwoValues,
                            recenter: bool = True,
                            ):
        """"Plot radial profile after optional recenter"""
        props = self._config['tool']['radial_profile']
        mark = self._config['mark'][props['mark_exam']]

        # prepare data
        rpx, rpy, xy = pa.radial_profile(self._data_2d, cnt_xy=xy,
                                         stamp_rad=self._pix_from_percent(self._config['stamp']),
                                         recenter=recenter)
        # plot data
        self.clear_exam()
        self._axes_exam.plot(rpx, rpy, mark['marker'],
                             color=mark['color'], alpha=mark['alpha'])
        pa.set_plot_props(self._axes_exam,
                          title="radial profile",
                          xlabel="distance to center",
                          ylabel=f'center at ({xy[0]:.1f}, {xy[1]:.1f})')
        self._axes_exam.figure.canvas.draw_idle()

        # mark temporal
        self._mark_temporal_in_2d(self._config['mark'][props['mark_data']], mark=xy)

    def plot_vprojection_exam_2d(self, xy):
        props = self._config['tool']['projection']

        self.plot_projection_exam_2d(xy[0], xy[1],
                                     props['width'],
                                     self._pix_from_percent(props, 'length_'),
                                     props['combine_op'],
                                     1)

    def plot_hprojection_exam_2d(self, xy):
        props = self._config['tool']['projection']

        self.plot_projection_exam_2d(xy[0], xy[1],
                                     self._pix_from_percent(props, 'length_'),
                                     props['width'],
                                     props['combine_op'],
                                     0)

    def plot_projection_exam_2d(self, x, y, lx, ly, op, op_row):
        props = self._config['tool']['projection']
        mark = self._config['mark'][props['mark_exam']]

        labels = ['horizontal', 'vertical']
        proj_label = ['column', 'row']

        x = int(x) - lx//2
        y = int(y) - ly//2
        if op_row == 0:
            index = np.arange(x, x+lx)
            width = ly
        else:
            index = np.arange(y, y+ly)
            width = lx
        region = self._data_2d[y:y+ly, x:x+lx]

        projection = getattr(region, op)(axis=op_row)
        if width > 1:
            ylabel = f"{op} of {width} {proj_label[1-op_row]}s around {x if op_row else y:d}"
        else:
            ylabel = f"{'column' if op_row else 'row'}  {x if op_row else y:d}"
        self.clear_exam()
        self._axes_exam.plot(index, projection, mark['marker'],
                             color=mark['color'], alpha=mark['alpha'],
                             )
        pa.set_plot_props(self._axes_exam,
                          title=f"{labels[op_row]} projection",
                          xlabel=f"{proj_label[op_row]}",
                          ylabel=ylabel)

        self._mark_temporal_in_2d(self._config['mark'][props['mark_data']],
                                  box=[x, x+lx, y, y+ly],
                                  )

    def mark_2d(self,
                xy: TwoValues,
                recenter: bool = True,
                decimals: int = 1,
                mark_type: str = 'point',
                ):
        """Add an extra mark at position. Store in list and draw in canvas.

        Parameters
        ----------
        xy:
           Position at which to add the mark
        recenter:
           whether to do a centroid centering
        decimals:
           number of decimals to keep
        mark_type:
            Type of mark to add. Currently, only 'point' is implemented
        """
        if recenter:
            xy = pa.subcentroid_xy(self._data_2d, xy, self._pix_from_percent(self._config['stamp']))
        xy = (round(xy[0], decimals), round(xy[1], decimals))

        if mark_type != "point":
            raise NotImplementedError(f"Invalid value '{mark_type}':"
                                      f" only 'point' is currently implemented as mark type")
        self._marks[mark_type].append(xy)
        self.draw_mark(mark_type)

    # noinspection PyUnusedLocal
    def delete_marks_from_2d(self,
                             xy: Optional[TwoValues],
                             clear_data: bool = True,
                             only: Optional[str] = None,
                             ):
        if only is None:
            for m in self._marks.keys():
                self.delete_marks_from_2d(xy, clear_data=clear_data, only=m)
            return

        for p in self._mark_patches[only]:
            p.remove()

        if clear_data:
            self._marks[only] = []

        self._axes_2d.figure.canvas.draw_idle()

    # noinspection PyUnusedLocal
    def vertical_flip_2d(self,
                         xy: TwoValues,
                         ):
        self._axes_2d.set_ylim(list(self._axes_2d.get_ylim())[::-1])
        self._axes_2d.figure.canvas.draw_idle()

    # noinspection PyUnusedLocal
    def horizontal_flip_2d(self,
                           xy: TwoValues,
                           ):
        self._axes_2d.set_xlim(list(self._axes_2d.get_xlim())[::-1])
        self._axes_2d.figure.canvas.draw_idle()

    # noinspection PyUnusedLocal
    def rotate_2d(self,
                  xy: TwoValues,
                  angle: float = 0,
                  ):
        image = self._axes_2d.get_images()[0]
        delta = self._data_2d.shape[1] - self._data_2d.shape[0]
        identity = transforms.Affine2D()
        trans_data = identity.rotate_deg_around(self._data_2d.shape[1]/2,
                                                self._data_2d.shape[0]/2, angle).translate(-delta/2, delta/2)

        self._extra_transform_data = trans_data
        trans = trans_data + image.get_transform()
        self._axes_2d.get_images()[0].set_transform(trans)
        self.clear_exam(temporal=True)

        self._axes_2d.figure.canvas.draw_idle()

    # noinspection PyUnusedLocal
    def terminate(self,
                  xy: TwoValues,
                  close_plot: bool = True,
                  verbose=True,
                  ):
        self.disconnect(verbose=verbose, close_plot=close_plot)
        if close_plot:
            if self._axes_exam is not None:
                plt.close(self._axes_exam.figure.number)
            plt.close(self._axes_2d.figure.number)

    # noinspection PyUnusedLocal
    def save_config(self,
                    xy: TwoValues,
                    ):
        self._save_config()

    # noinspection PyUnusedLocal
    def load_config(self,
                    xy: TwoValues,
                    reset: bool = False):
        self._load_config(reset=reset)

    ####################
    # Functions for handling hotkey options
    #
    #####################

    @property
    def options_last(self):
        """return the key of last option called"""
        return self._options_history[-1]

    def options_add(self, key, doc, fcn,
                    kwargs: dict = None,
                    ret: Optional[str] = None,
                    valid_in: Union[list[matplotlib.axes.Axes],
                                    matplotlib.axes.Axes, matplotlib.figure.Figure] = None,
                    ):
        """Add a new hotkey binding, it can overwrite only those given by default in .options_reset_config()

        Parameters
        ----------
        valid_in : list, matplotlib.axes.Axes, matplotlib.figure.Figure, optional
          Axes or Figures were the kwy is valid.  If None, then it is only valid in the datamap axes
        key: str
          Single character string
        doc: str
          Documentation for the hotkey
        fcn: str, callable
          Function from BindingsFunctions to be called if str; otherwise, a callable static function that
           receives event
        kwargs: dict
          Keyword arguments for to use in fcn() beyond the Event
        ret: str, optional
          Attribute of self defined in child class, return from fcn will be appended to this list
        """
        if kwargs is None:
            kwargs = {}

        if self._key_options is None:
            self.options_reset()

        if valid_in is None:
            valid_in = [self._axes_2d]
        elif isinstance(valid_in, matplotlib.figure.Figure):
            valid_in = valid_in.axes
        elif isinstance(valid_in, matplotlib.axes.Axes):
            valid_in = [valid_in]
        elif not isinstance(valid_in, (list, tuple)):
            raise TypeError(f"Invalid parameter valid_in ({valid_in})")

        if isinstance(fcn, str):
            fcn = getattr(self, fcn, None)

        if key not in self._key_options:
            self._key_options[key] = []
        for ax in valid_in:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise TypeError(f"valid_in element list of incorrect type ({ax}), it can only be an axes")
            if ax in [x[4] for x in self._key_options[key]]:
                raise ValueError(f"Key '{key}' has already been added for '{doc}' in axes {ax}, "
                                 f"cannot add it for '{doc}'")
            self._key_options[key].append((doc, fcn, kwargs, ret, ax))

    def options_help(self,
                     axes: matplotlib.axes.Axes = None,
                     fontsize: int = 9,
                     ):
        yy = 0.91

        txt = "Available hotkeys:"
        if axes is None:
            print(txt)
        else:
            axes.text(0.1, 0.95, txt,
                      fontsize=fontsize+1, transform=axes.transAxes)
        for key, options in self._key_options.items():
            docs = []
            for (doc, fcn, kwargs, ret, valid_in) in options:
                docs.append(doc)
            txt = f"+ {key}: {'/'.join(set(docs))}"
            if axes is None:
                print(txt)
            else:
                axes.text(0.1, yy, txt,
                          fontsize=fontsize, transform=axes.transAxes)
                yy -= 0.03

    # noinspection PyUnusedLocal
    def _options_help(self,
                      event: KeyEvent,
                      axes: matplotlib.axes.Axes = None,
                      ):
        self.options_help(axes=axes)

    def _options_choose(self,
                        event: KeyEvent):
        for key, options in self._key_options.items():
            if event.key == key:
                for (doc, fcn, kwargs, ret, valid_in) in options:
                    if event.inaxes == valid_in:
                        break
                else:
                    return

                kwargs = kwargs.copy()
                inverse_transform = self._axes_2d.get_images()[0].get_transform().inverted()
                xy = kwargs.pop('xy', inverse_transform.transform((event.x, event.y)))

                if ret is None:
                    fcn(xy, **kwargs)
                else:
                    getattr(self, ret).append(fcn(xy, **kwargs))

                self._options_history.append(key)

                return

    def options_reset(self, config_options=True, help_option=True, quit_option=True, valid_in=None):
        """Loads config file and add default config saving/loading options

        Parameters
        ----------
        valid_in: axes, figure
            specify the validity for the help and config options, the default None indicates the whole
            figure that has the datamap
        quit_option
            Whether to include the quit option 'q'
        help_option: bool
            Whether to include the help option '?'
        config_options : bool
            Whether to include the config options 'L'oad, 'S'ave, load-'F'actory-defaults
        """
        file_exists = self._load_config()
        self._key_options = {}
        if valid_in is None:
            valid_in = self._axes_2d.figure

        if help_option:
            self.options_add('?', "hotkey help", "_options_help", valid_in=valid_in)
        if config_options:
            if not file_exists:
                raise FileNotFoundError("Config file for interactive not found. "
                                        "Use procastro.BindingFunctions.options_reset(config_options=False) "
                                        "to avoid this warning")
            self.options_add('L', f"reload configuration from '{pa.user_confdir(self._config_file)}'",
                             'load_config', valid_in=valid_in)
            self.options_add('S', 'save configuration', 'save_config',
                             valid_in=valid_in)
            self.options_add('F', 'load default configuration from factory', 'load_config',
                             kwargs={'reset': True},
                             valid_in=valid_in)
        if quit_option:
            self.options_add('q', 'terminate interactive imshow', 'terminate', valid_in=valid_in)


class BindingsImshowz(BindingsFunctions):

    def __init__(self,
                 image: np.ndarray,
                 axes_data: matplotlib.axes.Axes,
                 axes_exam: matplotlib.axes.Axes = None,
                 colorbar_data: matplotlib.axes.Axes = None,
                 colorbar_exam: matplotlib.axes.Axes = None,
                 config_file: str = "interactive.toml",
                 ):

        super(BindingsImshowz, self).__init__(axes_data, axes_exam,
                                              config_file=config_file,
                                              cb_data=colorbar_data,
                                              cb_exam=colorbar_exam,
                                              title="ProcAstro's interactive imshowz()",
                                              )

        self.options_reset()

        self.options_add('r', 'radial profile', 'plot_radial_exam_2d')
        self.options_add('9', 'zoom with radius 9', 'zoom_exam_2d',
                         kwargs={'scale': 'minmax', 'stamp_rad': 9})
        self.options_add('5', 'zoom with radius 5', 'zoom_exam_2d',
                         kwargs={'scale': 'minmax', 'stamp_rad': 5, 'text': True})
        self.options_add('z', 'zoom into stamp', 'zoom_exam_2d')
        self.options_add('9', 'zoom with radius 9 at same point', 'zoom_exam_2d',
                         kwargs={'xy': None, 'scale': None, 'text': False, 'stamp_rad': 9},
                         valid_in=self.axes_exam)
        self.options_add('5', 'zoom with radius 5 at same point', 'zoom_exam_2d',
                         kwargs={'xy': None, 'scale': None, 'text': True, 'stamp_rad': 5},
                         valid_in=self.axes_exam)
        self.options_add('z', 'zoom with stamp radius at same point', 'zoom_exam_2d',
                         kwargs={'xy': None, 'scale': None},
                         valid_in=self.axes_exam)
        self.options_add('s', 'Cycle scale for data map', 'change_scale_data',
                         kwargs={'scale': 'cycle'},
                         valid_in=self.axes_data)
        self.options_add('s', 'Cycle contrast scale for examination map', 'zoom_exam_2d',
                         kwargs={'scale': 'cycle', 'stamp_rad': None, 'text': None, 'xy': None},
                         valid_in=self.axes_exam)
        self.options_add('w', 'Rotate 90 degrees counter-clockwise', 'rotate_2d',
                         kwargs={'angle': 90})
        self.options_add('e', 'Rotate 90 degrees clockwise', 'rotate_2d',
                         kwargs={'angle': -90})
        self.options_add('h', 'horizontal projection', 'plot_hprojection_exam_2d')
        self.options_add('v', 'vertical projection', 'plot_vprojection_exam_2d')
        self.options_add('m', 'mark a new position', 'mark_2d',
                         kwargs={'recenter': False,
                                 'decimals': self._config['tool']['marking']['rounding_user'],
                                 },
                         )
        self.options_add('c', 'mark a new position after centering', 'mark_2d',
                         kwargs={'recenter': True,
                                 'decimals': self._config['tool']['marking']['rounding_centroid'],
                                 },
                         )
        self.options_add('d', 'delete all marks', 'delete_marks_from_2d',)
        self.options_add('x', 'flip X-axis', 'horizontal_flip_2d')
        self.options_add('y', 'flip Y-axis', 'vertical_flip_2d')

        self.set_data_2d(image, scale='zscale')
        self.options_help(axes=self.axes_exam, fontsize=10)

        self.connect()

    def get_marks(self):
        return self._marks['point']
