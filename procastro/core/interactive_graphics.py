import os.path
from pathlib import Path

import matplotlib
matplotlib.rcParams['keymap.xscale'].remove('L')
matplotlib.rcParams['keymap.quit'].remove('q')
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np
import procastro as pa
from functools import wraps as _wraps
import configparser as cp
import re

from typing import Optional, Callable, Any
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
    def __init__(self,
                 config_file = "interactive.cfg"):
        self._config_file = config_file
        self._key_options = None
        self._axes_aux = None
        self._axes_2d = None
        self._data_2d = None

        self._config = cp.ConfigParser()
        self._temporal_artist = []
        self._mark_patches = []
        self._cid = {}

        self._stamp_mark = None
        self._stamp_radial = None
        self._projection_length = None

    def set_axes_2d(self, ax):
        self._axes_2d = ax

    def set_axes_aux(self, ax):
        self._axes_aux = ax

    def set_data_2d(self, data):
        self._data_2d = data
        self._stamp_mark = self._rad_from_pix_percent('mark')
        self._stamp_radial = self._rad_from_pix_percent('radial_profile')
        self._projection_length = self._rad_from_pix_percent('projection', prop='length')

    def _rad_from_pix_percent(self, param, prop='stamp'):
        if self._config[param].getint(f'{prop}_pix') == 0:
            yy, xx = self._data_2d.shape
            return int(self._config[param].getint(f'{prop}_percent') * max(xx, yy) / 200)
        return self._config[param].getint(f'{prop}_pix')

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
            self._temporal_artist.extend(self._axes_2d.plot(mark[0], mark[1], 'x',
                                                            color=props['mark_color'],
                                                            alpha=props.getfloat('mark_alpha'),
                                                            )
                                         )
        if box is not None:
            self._temporal_artist.append(self._axes_2d.fill_between(box[0:2], [box[2]] * 2, y2=[box[3]] * 2,
                                                                    color=props['mark_color'],
                                                                    alpha=props.getfloat('mark_alpha'),
                                                                    )
                                         )

        self._axes_2d.figure.canvas.draw_idle()

    def _load_config(self,
                     reset: bool = False):
        if reset:
            self._config.read(pa.default_for_procastro_dir(self._config_file))
            print("Forced reset: loaded factory defaults")
        else:
            file = pa.file_from_procastro_dir(self._config_file)
            self._config.read(file)
            print(f"Loaded configuration from: {file}")

    def _save_config(self,
                     ):
        file = pa.file_from_procastro_dir(self._config_file)
        with open(file, 'w') as configfile:
            self._config.write(configfile)
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

    def connect(self, axes_2d, axes_aux):
        self._axes_2d = axes_2d
        self._axes_aux = axes_aux
        self._cid['key_press'] = axes_2d.figure.canvas.mpl_connect('key_press_event', self._on_key_press)
        axes_2d.figure.canvas.draw_idle()
        axes_aux.figure.show()

        print("Entering interactive mode ('?' for help)")
        axes_2d.figure.canvas.start_event_loop()

    def _on_key_press(self, event):
        self._options_choose(event)

    ############################
    # Callable functions
    #
    ############################

    def plot_radial_from_2d(self,
                            event,
                            recenter: bool = True,
                            ):
        """"Plot radial profile after optional recenter"""
        cxy = (event.xdata, event.ydata)
        props = self._config['radial_profile']

        # prepare data
        rpx, rpy, cxy = pa.radial_profile(self._data_2d, cnt_xy=cxy,
                                          stamp_rad=self._stamp_radial,
                                          recenter=recenter)
        # plot data
        self._axes_aux.cla()
        self._axes_aux.plot(rpx, rpy, props['marker'],
                            color=props['color'], alpha=props.getfloat('alpha'))
        pa.set_plot_props(self._axes_aux,
                          title="radial profile",
                          xlabel="distance to center",
                          ylabel=f'center at ({cxy[0]:.1f}, {cxy[1]:.1f})')
        self._axes_aux.figure.canvas.draw_idle()

        # mark temporal
        self._mark_temporal_in_2d(props, mark=cxy)

    def plot_vprojection_from_2d(self, event):
        cxy = (event.xdata, event.ydata)
        props = self._config['projection']
        self.plot_projection_from_2d(cxy[0], cxy[1],
                                     props.getint('width'),
                                     self._projection_length,
                                     props['combine_op'],
                                     1)

    def plot_hprojection_from_2d(self, event):
        cxy = (event.xdata, event.ydata)
        props = self._config['projection']
        self.plot_projection_from_2d(cxy[0], cxy[1],
                                     self._projection_length,
                                     props.getint('width'),
                                     props['combine_op'],
                                     0)

    def plot_projection_from_2d(self, x, y, lx, ly, op, op_row):
        props = self._config['projection']
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
        self._axes_aux.cla()
        self._axes_aux.plot(index, projection, props['marker'],
                            color=props['color'], alpha=props.getfloat('alpha'),
                            )
        pa.set_plot_props(self._axes_aux,
                          title=f"{labels[op_row]} projection",
                          xlabel=f"{proj_label[op_row]}",
                          ylabel=ylabel)

        self._mark_temporal_in_2d(props,
                                  box=[x, x+lx, y, y+ly],
                                  )

    def return_mark_from_2d(self, event, marks=None, recenter=True, decimals=None):
        cxy = (event.xdata, event.ydata)
        props = self._config['mark']

        yy, xx = self._data_2d.shape
        if recenter:
            cxy = pa.subcentroid_xy(self._data_2d, cxy, self._stamp_mark)

        radius = props.getint('radius') * max(xx, yy) / 100
        if decimals is None:
            decimals = props.getint('rounding_user')

        xy = (round(cxy[0], decimals), round(cxy[1], decimals))

        colors = _to_str_tuple(props['color'])

        idx = len(getattr(self, marks))
        color = colors[idx] if idx < len(colors) else colors[-1]

        circ = patches.Circle(xy,
                              radius=radius,
                              alpha=props.getfloat('mark_alpha'),
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
        plt.close(self._axes_aux.figure.number)
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
        if key in self._key_options.keys():
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
        self._load_config()

        self._key_options = {}

        self.options_add('?', "Hotkey help", "_options_help", {}, None)

        self.options_add('L', 'load saved configuration', 'load_config',
                         {}, None)
        self.options_add('S', 'save configuration', 'save_config',
                         {}, None)
        self.options_add('F', 'load default configuration from factory', 'load_config',
                         {'reset': True}, None)


class BindingsImshowz(BindingsFunctions):

    def __init__(self,
                 image: np.ndarray,
                 axes: matplotlib.figure.Figure,
                 config_file: str = "interactive.cfg",
                 ):
        super(BindingsImshowz, self).__init__()

        self._marks_xy = []

        self.options_reset_config()

        self.options_add('r', 'radial prbofile', 'plot_radial_from_2d',
                         {}, None)
        self.options_add('h', 'horizontal projection', 'plot_hprojection_from_2d',
                         {}, None)
        self.options_add('v', 'vertical projection', 'plot_vprojection_from_2d',
                         {}, None)
        self.options_add('m', 'mark a new position', 'return_mark_from_2d',
                         {'recenter': False, 'marks': '_marks_xy',
                          'decimals': self._config['mark'].getint('rounding_user'),
                          },
                         '_marks_xy')
        self.options_add('c', 'mark a new position after centering', 'return_mark_from_2d',
                         {'recenter': True, 'marks': '_marks_xy',
                          'decimals': self._config['mark'].getint('rounding_centroid'),
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

        f, ax = pa.figaxes()

        self.connect(axes, ax)

    def get_marks(self):
        return self._marks_xy
