import os.path
from pathlib import Path

import matplotlib
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np
import procastro as pa
from functools import wraps as _wraps
import re

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml
import tomli_w
from typing import Optional, Callable, Any

matplotlib.rcParams['keymap.xscale'].remove('L')
matplotlib.rcParams['keymap.quit'].remove('q')

TwoValues = tuple[float, float]
FunctionArgs = tuple[Callable, list[Any]]
FunctionArgsKw = tuple[Callable, list[Any], dict]


def _to_str_tuple(value):
    value = re.sub(' *, *', ',', value)
    if ',' in value:
        return value.split(',')

    return value,


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
    def __init__(self, axes_data,
                 axes_exam,
                 config_file,
                 cb_data=None,
                 cb_exam=None,
                 ):
        self._config_file = config_file
        self._key_options = None
        self._data_2d = None

        self._config = {}
        self._temporal_artist = []
        self._mark_patches = []
        self._cid = {}

        self._axes_2d = axes_data
        self._axes_exam = axes_exam
        self._colorbar_data = cb_data
        self._colorbar_exam = cb_exam

    def set_data_2d(self,
                    data: np.ndarray):
        self._data_2d = data
        if self._colorbar_data is not None:
            fig = self._axes_2d.figure
            fig.colorbar(self._axes_2d.get_images()[0],
                         cax=self._colorbar_data,
                         orientation="horizontal",
                         # extend='both',
                         )

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

        if self._axes_2d is None:
            return
        if self._temporal_artist:
            while True:
                try:
                    t = self._temporal_artist.pop()
                except IndexError:
                    break
                t.remove()
        if mark is not None:
            self._temporal_artist.extend(self._axes_2d.plot(mark[0], mark[1], props['marker'],
                                                            color=props['color'],
                                                            alpha=props['alpha'],
                                                            )
                                         )
        if box is not None:
            self._temporal_artist.append(self._axes_2d.fill_between(box[0:2], [box[2]] * 2, y2=[box[3]] * 2,
                                                                    color=props['color'],
                                                                    alpha=props['alpha'],
                                                                    )
                                         )

        self._axes_2d.figure.canvas.draw_idle()

    def _load_config(self,
                     reset: bool = False):

        def update_vals(new_config, old_config):
            for key, item in new_config.items():
                if isinstance(item, dict):
                    item = update_vals(item, old_config[key])
                old_config[key] = item
            return old_config

        self._config = toml.loads(Path(pa.default_for_procastro_dir(self._config_file)).read_text(encoding='utf-8'))
        if reset:
            print("Forced reset: loaded factory defaults")
        else:
            file = pa.file_from_procastro_dir(self._config_file)
            try:
                new = toml.loads(Path(file).read_text(encoding='utf-8'))
                self._config = update_vals(new, self._config)
                print(f"Loaded configuration from: {file}")
            except toml.TOMLDecodeError:
                print(f"Skipping configuration from corrupt local config ({file}). "
                      f"It is recommended to save new version.")

    def _save_config(self,
                     ):
        file = pa.file_from_procastro_dir(self._config_file)
        with open(file, 'wb') as fp:
            tomli_w.dump(self._config, fp)
        print(f"Saved configuration to: {file}")

    ############################
    # start-end interactive mode
    #
    ############################

    def disconnect(self):
        print("... exiting interactive mode")
        self._axes_2d.figure.canvas.stop_event_loop()
        for cid in self._cid.values():
            self._axes_2d.figure.canvas.mpl_disconnect(cid)

    def connect(self):
        self._cid['key_press'] = self._axes_2d.figure.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._axes_2d.figure.canvas.draw_idle()
        self._axes_exam.figure.canvas.draw_idle()

        print("Entering interactive mode ('?' for help)")
        self._axes_2d.figure.canvas.start_event_loop()

    def _on_key_press(self, event):
        if event.inaxes == self._axes_2d:
            self._options_choose(event)

    ############################
    # Callable functions
    #
    ############################

    def clear_exam(self):
        """Clear examination area for a new plot, and its colorbar if there is any"""
        axes = [self._axes_exam]
        if self._colorbar_exam is not None:
            # ims = self._axes_exam.get_images()
            # if len(ims)>0:
            #     ims[0].colorbar.remove()
            axes.append(self._colorbar_exam)
        for ax in axes:
            ax.cla()
            ax.set_aspect("auto")
            ax.yaxis.tick_right()

    def zoom_stamp_2d(self,
                      event,
                      scale: str = 'original',
                      text: bool = False,
                      stamp_rad: int = 0,
                      ):
        """"returns a zoom of the stamp"""

        self.clear_exam()

        cxy = (int(event.xdata), int(event.ydata))
        props = self._config['tool']['zoom']

        if stamp_rad == 0:
            stamp_rad = self._pix_from_percent(self._config['stamp'])

        extents = cxy[0] - stamp_rad, cxy[0] + stamp_rad, cxy[1] - stamp_rad, cxy[1] + stamp_rad
        stamp = self._data_2d[extents[2]:extents[3], extents[0]:extents[1]]

        min_value = np.min(stamp)
        max_value = np.max(stamp)
        min_idxs = np.transpose((stamp == min_value).nonzero())
        max_idxs = np.transpose((stamp == max_value).nonzero())
        mean_value = np.mean(stamp)
        median_value = np.median(stamp)

        if scale == 'zscale':
            vmin, vmax = pa.zscale(self._data_2d, trim=0.02)
        elif scale == 'linear':
            vmin, vmax = min_value, max_value
        elif scale == 'original':
            vmin, vmax = self._axes_2d.get_images()[0].get_clim()
        else:
            raise ValueError(f"zoom called with an unsupported 'scale' option: {scale}")
        mid = (vmin+vmax)/2
        im = self._axes_exam.imshow(stamp,
                                    cmap=props['colormap'],
                                    extent=extents, vmin=vmin, vmax=vmax,
                                    origin='lower',
                                    )

        stat_mark = props['stat']
        border = stat_mark['from_border']
        if self._colorbar_exam is not None:
            ending = 0
            if vmax != max_value:
                ending |= 2
            if vmin != min_value:
                ending |= 1
            self._axes_exam.figure.colorbar(im, cax=self._colorbar_exam,
                                            orientation='horizontal',
                                            # extend=['neither', 'min', 'max', 'both'][ending],
                                            )

            mxv, mxl = [max_value, ''] if max_value == vmax else [vmax, " ->"]
            mnv, mnl = [min_value, ''] if min_value == vmin else [vmin, "<- "]
            for value, vert, sym, label, ha in zip([mxv, mnv, median_value, mean_value],
                                                   [0.5, 0.5, 0.2, 0.8],
                                                   ['|', '|', 'd', 'd'],
                                                   [f'max {max_value} {mxl}', f'{mnl} {min_value} min',
                                                    f'median {median_value}', f'mean {mean_value:.1f}'],
                                                   ['right'] + ['left']*3,
                                                   ):
                color = props['facecolor_dark' if value > mid else 'facecolor_light']
                self._colorbar_exam.plot(value, vert, sym,
                                         color=color)
                self._colorbar_exam.text(value, vert, label,
                                         ha=ha, va='center',
                                         color=color)

        rectangle = patches.Rectangle
        color = stat_mark['color']
        ax = self._axes_exam
        for idx, label in [[min_idxs, "min"], [max_idxs, "max"]]:
            for y, x in idx:
                y += extents[2]
                x += extents[0]
                ax.add_patch(rectangle((x + border, y + border),
                                       1 - 2*border, 1 - 2*border,
                                       alpha=stat_mark['alpha'], linewidth=stat_mark['linewidth'],
                                       color=color, fill=False)
                             )
                ax.text(x + 2*border, y, label, ha="left", va="bottom", color=color)

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
                        )

        pa.set_plot_props(self._axes_exam,
                          title=f"max:{max_value} * min:{min_value} * mean:{mean_value:.1f} * median:{median_value}",
                          ylabel=f"{scale[0].upper()}{scale[1:]} scale",
                          )

        self._mark_temporal_in_2d(self._config['mark'][props['mark_data']],
                                  box=extents,
                                  )

    def plot_radial_from_2d(self,
                            event,
                            recenter: bool = True,
                            ):
        """"Plot radial profile after optional recenter"""
        cxy = (event.xdata, event.ydata)
        props = self._config['tool']['radial_profile']
        mark = self._config['mark'][props['mark_exam']]

        # prepare data
        rpx, rpy, cxy = pa.radial_profile(self._data_2d, cnt_xy=cxy,
                                          stamp_rad=self._pix_from_percent(self._config['stamp']),
                                          recenter=recenter)
        # plot data
        self.clear_exam()
        self._axes_exam.plot(rpx, rpy, mark['marker'],
                             color=mark['color'], alpha=mark['alpha'])
        pa.set_plot_props(self._axes_exam,
                          title="radial profile",
                          xlabel="distance to center",
                          ylabel=f'center at ({cxy[0]:.1f}, {cxy[1]:.1f})')
        self._axes_exam.figure.canvas.draw_idle()

        # mark temporal
        self._mark_temporal_in_2d(self._config['mark'][props['mark_data']], mark=cxy)

    def plot_vprojection_from_2d(self, event):
        cxy = (event.xdata, event.ydata)
        props = self._config['tool']['projection']

        self.plot_projection_from_2d(cxy[0], cxy[1],
                                     props['width'],
                                     self._pix_from_percent(props, 'length_'),
                                     props['combine_op'],
                                     1)

    def plot_hprojection_from_2d(self, event):
        cxy = (event.xdata, event.ydata)
        props = self._config['tool']['projection']

        self.plot_projection_from_2d(cxy[0], cxy[1],
                                     self._pix_from_percent(props, 'length_'),
                                     props['width'],
                                     props['combine_op'],
                                     0)

    def plot_projection_from_2d(self, x, y, lx, ly, op, op_row):
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

    def return_mark_from_2d(self, event, marks=None, recenter=True, decimals=None):
        cxy = (event.xdata, event.ydata)
        props = self._config['tool']['marking']

        yy, xx = self._data_2d.shape
        if recenter:
            cxy = pa.subcentroid_xy(self._data_2d, cxy, self._pix_from_percent(self._config['stamp']))

        radius = props['radius'] * max(xx, yy) / 100
        if decimals is None:
            decimals = props['rounding_user']

        xy = (round(cxy[0], decimals), round(cxy[1], decimals))

        colors = _to_str_tuple(props['color'])

        idx = len(getattr(self, marks))
        color = colors[idx] if idx < len(colors) else colors[-1]

        circ = patches.Circle(xy,
                              radius=radius,
                              alpha=props['alpha'],
                              color=color)
        self._mark_patches.append(circ)
        self._axes_2d.add_patch(circ)
        self._axes_2d.figure.canvas.draw_idle()

        return xy

    # noinspection PyUnusedLocal
    def delete_marks_from_2d(self,
                             event: KeyEvent,
                             field: str = ''):
        for p in self._mark_patches:
            p.remove()
        setattr(self, field, [])
        self._mark_patches = []
        self._axes_2d.figure.canvas.draw_idle()

    # noinspection PyUnusedLocal
    def vertical_flip_2d(self,
                         event: KeyEvent,
                         ):
        self._axes_2d.set_ylim(list(self._axes_2d.get_ylim())[::-1])
        self._axes_2d.figure.canvas.draw_idle()

    # noinspection PyUnusedLocal
    def horizontal_flip_2d(self,
                           event: KeyEvent,
                           ):
        self._axes_2d.set_xlim(list(self._axes_2d.get_xlim())[::-1])
        self._axes_2d.figure.canvas.draw_idle()

    # noinspection PyUnusedLocal
    def terminate(self,
                  event: KeyEvent):
        self.disconnect()
        plt.close(self._axes_exam.figure.number)
        plt.close(self._axes_2d.figure.number)

    # noinspection PyUnusedLocal
    def save_config(self,
                    event: KeyEvent,
                    ):
        self._save_config()

    # noinspection PyUnusedLocal
    def load_config(self,
                    event: KeyEvent,
                    reset: bool = False):
        self._load_config(reset=reset)

    ####################
    # Functions for handling hotkey options
    #
    #####################

    def options_add(self, key, doc, fcn, kwargs, ret):
        """Add a new hotkey binding, it can overwrite only those given by default in .options_reset_config()

        Parameters
        ----------
        key: str
          Single character string
        doc: str
          Documentation for the hotkey
        fcn: str
          Function from BindingsFunctions to be called
        kwargs: dict
          Keyword arguments for to use in fcn() beyond the Event
        ret:
          Attribute of self defined in child class, typically used to return a value
        """
        overwritable = ['S', 'L', 'F', '?']

        if key in self._key_options.keys() and key not in overwritable:
            raise ValueError(f"Key '{key}' has already been added for '{doc}', cannot add it for '{doc}'")
        self._key_options[key] = (doc, fcn, kwargs, ret)

    # noinspection PyUnusedLocal
    def _options_help(self,
                      event: KeyEvent):
        print("Available hotkeys:")
        for key, (doc, fcn, kwargs, ret) in self._key_options.items():
            print(f"+ {key}: {doc}")

    def _options_choose(self,
                        event: KeyEvent):
        for key, (doc, fcn, kwargs, ret) in self._key_options.items():
            if event.key == key:
                if ret is None:
                    getattr(self, fcn)(event, **kwargs)
                else:
                    getattr(self, ret).append(getattr(self, fcn)(event, **kwargs))

    def options_reset_config(self):
        """Loads config file and add default config saving/loading options

        """
        self._load_config()

        self._key_options = {}
        self.options_add('?', "hotkey help", "_options_help", {}, None)
        self.options_add('L', f"reload configuration from '{pa.file_from_procastro_dir(self._config_file)}'",
                         'load_config',
                         {}, None)
        self.options_add('S', 'save configuration', 'save_config',
                         {}, None)
        self.options_add('F', 'load default configuration from factory', 'load_config',
                         {'reset': True}, None)


class BindingsImshowz(BindingsFunctions):

    def __init__(self,
                 image: np.ndarray,
                 axes_data: matplotlib.axes.Axes,
                 axes_exam: matplotlib.axes.Axes = None,
                 colorbar_data: matplotlib.axes.Axes = None,
                 colorbar_exam: matplotlib.axes.Axes = None,
                 config_file: str = "interactive.toml",
                 ):

        f, axes_exam = pa.figaxes(axes_exam)
        axes_exam.yaxis.tick_right()
        if colorbar_exam is not None:
            colorbar_exam.yaxis.tick_right()
        f.show()

        super(BindingsImshowz, self).__init__(axes_data, axes_exam,
                                              config_file,
                                              cb_data=colorbar_data,
                                              cb_exam=colorbar_exam,
                                              )

        self.options_reset_config()

        self._marks_xy = []

        self.options_add('r', 'radial profile', 'plot_radial_from_2d',
                         {}, None)
        self.options_add('9', 'zoom with radius 9', 'zoom_stamp_2d',
                         {'scale': 'linear', 'stamp_rad': 9}, None)
        self.options_add('5', 'zoom with radius 5', 'zoom_stamp_2d',
                         {'scale': 'linear', 'stamp_rad': 5, 'text': True}, None)
        self.options_add('z', 'zoom into stamp', 'zoom_stamp_2d',
                         {'scale': 'original'}, None)
        self.options_add('Z', 'zoom into stamp, recomputing scale by zscale in stamp', 'zoom_stamp_2d',
                         {'scale': 'zscale'}, None)
        self.options_add('X', 'zoom into stamp, recomputing scale linearly in the stamp', 'zoom_stamp_2d',
                         {'scale': 'linear'}, None)
        self.options_add('h', 'horizontal projection', 'plot_hprojection_from_2d',
                         {}, None)
        self.options_add('v', 'vertical projection', 'plot_vprojection_from_2d',
                         {}, None)
        self.options_add('m', 'mark a new position', 'return_mark_from_2d',
                         {'recenter': False, 'marks': '_marks_xy',
                          'decimals': self._config['tool']['marking']['rounding_user'],
                          },
                         '_marks_xy')
        self.options_add('c', 'mark a new position after centering', 'return_mark_from_2d',
                         {'recenter': True, 'marks': '_marks_xy',
                          'decimals': self._config['tool']['marking']['rounding_centroid'],
                          },
                         '_marks_xy')
        self.options_add('d', 'delete all marks', 'delete_marks_from_2d',
                         {'field': '_marks_xy'},
                         None)
        self.options_add('x', 'flip X-axis', 'horizontal_flip_2d',
                         {}, None)
        self.options_add('y', 'flip Y-axis', 'vertical_flip_2d',
                         {}, None)
        self.options_add('q', 'terminate interactive imshow', 'terminate',
                         {}, None)

        self.set_data_2d(image)
        self.connect()

    def get_marks(self):
        return self._marks_xy
