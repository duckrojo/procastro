import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches
import numpy as np
import dataproc as pa

from typing import Optional, Tuple, Union


class _imshowz_binding:
    def set_radial_props(self,
                         color: str = 'red',
                         alpha: float = 0.5,
                         stamp_rad: int = 0,
                         stamp_percent: int = 5,
                         ):
        """

        Parameters
        ----------
        alpha
        color: str or (str, ...)
           colors of circle's mark if more marks than options, keeps repeating last one
        stamp_rad: int
           stamp radius to compute the centroid, if zero then compute from  stamp_percent
        stamp_percent: int
           size of stamp in percent of the whole image array, only use if stamp_rad==0
        """
        self._radial_props['color'] = color
        self._radial_props['alpha'] = alpha

        if stamp_rad == 0:
            xx, yy = self.image.shape
            stamp_rad = int(stamp_percent * max(xx, yy) / 200)
        self._radial_props['stamp_rad'] = stamp_rad

    def set_projection_props(self,
                             length_percent: int = 15,  # size of circle' radius in percent
                             length_pixels: int = 0,  # size of circle' radius in percent
                             color: str = 'red',
                             alpha: float = 0.5,
                             combine_op: str = 'sum',
                             width: int = 1,
                             ):
        """

        Parameters
        ----------
        alpha: float
           transparency
        length_percent: int
           length of projection in percentage
        length_pixels: int
           length of projection in pixels. If zero, then use length_percent
        width: int
           width of projection in pixels. values are cobined with combine_op
        color: str
           colors of projection mark
        combine_op: str
           operations for combination along width
        """
        self._projection_props['color'] = color
        self._projection_props['width'] = width
        self._projection_props['alpha'] = alpha
        self._projection_props['combine_op'] = combine_op
        if length_pixels == 0:
            xx, yy = self.image.shape
            length_pixels = int(length_percent * max(xx, yy) / 200)
        self._projection_props['length'] = length_pixels

    def set_centroid_props(self,
                           radius: int = 2,  # size of circle' radius in percent
                           color: Union[str, Tuple[str, ...]] = ('black', 'red'),
                           alpha: float = 0.3,  # alpha of mark
                           rounding: int = 1,  # number of decimals for rounding of mark coordinates
                           stamp_rad: int = 0,
                           stamp_percent: int = 5,
                           ):
        """

        Parameters
        ----------
        radius: int
           size of circle' radius in percent
        color: str or (str, ...)
           colors of circle's mark if more marks than options, keeps repeating last one
        alpha: float
           transparency
        rounding: int
           number of decimals for rounding of mark coordinates
        stamp_rad: int
           stamp radius to compute the centroid, if zero then compute from  stamp_percent
        stamp_percent: int
           size of stamp in percent of the whole image array, only use if stamp_rad==0
        """
        self._centroid_props['radius'] = radius
        if isinstance(color, str):
            color = (color,)
        self._centroid_props['color'] = color
        self._centroid_props['alpha'] = alpha
        self._centroid_props['rounding'] = rounding

        if stamp_rad == 0:
            xx, yy = self.image.shape
            stamp_rad = int(stamp_percent * max(xx, yy) / 200)
        self._centroid_props['stamp_rad'] = stamp_rad

    def set_mark_props(self,
                       radius: int = 2,  # size of circle' radius in percent
                       color: Union[str, Tuple[str, ...]] = ('black', 'red'),  # colors of circle's mark if more marks than options, keeps repeating last one
                       alpha: float = 0.3,  # alpha of mark
                       rounding: int = 0,   # number of decimals for rounding of mark coordinates
                       ):
        """

        Parameters
        ----------
        radius: int
           size of circle' radius in percent
        color: str or (str, ...)
           colors of circle's mark if more marks than options, keeps repeating last one
        alpha: float
           transparency
        rounding: int
           number of decimals for rounding of mark coordinates
        """
        self._mark_props['radius'] = radius
        if isinstance(color, str):
            color = (color, )
        self._mark_props['color'] = color
        self._mark_props['alpha'] = alpha
        self._mark_props['rounding'] = rounding

    def __init__(self,
                 image: np.ndarray,
                 axes: matplotlib.figure.Figure,
                 mark_props: Optional[dict] = None,
                 centroid_props: Optional[dict] = None,
                 radial_props: Optional[dict] = None,
                 projection_props: Optional[dict] = None,
                 ):

        f, ax = pa.figaxes()
        self.ax_extra = ax

        self.image = image
        self.axes = axes
        self.cid = {}
        self.outs = {'marks_xy': [],
                     }
        self._temporal_artist = None

        self._mark_props = {}
        self._centroid_props = {}
        self._radial_props = {}
        self._projection_props = {}

        if mark_props is None:
            mark_props = {}
        if centroid_props is None:
            centroid_props = {}
        if radial_props is None:
            radial_props = {}
        if projection_props is None:
            projection_props = {}

        self.set_mark_props(**mark_props)
        self.set_centroid_props(**centroid_props)
        self.set_radial_props(**radial_props)
        self.set_projection_props(**projection_props)

        self.connect()
        self.axes.figure.canvas.draw_idle()
        self.axes.figure.show()
        print("Entering interactive mode ('?' for help)")
        self.axes.figure.canvas.start_event_loop()

    def disconnect(self):
        print ("... exiting interactive mode")
        self.axes.figure.canvas.stop_event_loop()
        for cid in self.cid.values():
            self.axes.figure.canvas.mpl_disconnect(cid)

    def connect(self):
        self.cid['key_press'] = self.axes.figure.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        lims = self.axes.get_xlim()
        dx = lims[1] - lims[0]
        lims = self.axes.get_ylim()
        dy = lims[1] - lims[0]

        canvas = self.axes.figure.canvas

        def store_draw_mark(x, y, props):
            radius = props['radius'] * max(dx, dy) / 100
            colors = props['color']
            alpha = props['alpha']
            decimals = props['rounding']

            xy = (round(x, decimals), round(y, decimals))

            n_marks = len(self.outs['marks_xy'])
            color = colors[n_marks] if n_marks < len(colors) else colors[-1]
            self.outs['marks_xy'].append(xy)

            circ = patches.Circle(xy, radius=radius, alpha=alpha, color=color)
            self.axes.add_patch(circ)
            canvas.draw_idle()

        def draw_projection(x, y, lx, ly, op, op_row):
            x = int(x) - lx//2
            y = int(y) - ly//2
            if op_row == 0:
                index = np.arange(x, x+lx)
                width = ly
            else:
                index = np.arange(y, y+ly)
                width = lx
            region = self.image[y:y+ly, x:x+lx]

            projection = getattr(region, op)(axis=op_row)
            if width > 1:
                ylabel = f"{op} of {width} {'columns' if op_row else 'rows'} around {x if op_row else y:d}"
            else:
                ylabel = f"{'column' if op_row else 'row'}  {x if op_row else y:d}"
            self.ax_extra.cla()
            self.ax_extra.plot(index, projection)
            pa.set_plot_props(self.ax_extra,
                              title=f"{'vertical' if op_row else 'horizontal'} projection",
                              xlabel=f"{'row' if op_row else 'column'}",
                              ylabel=ylabel)

            if self._temporal_artist:
                for tmp in self._temporal_artist:
                    tmp.remove()
            self._temporal_artist = self.axes.fill_between([x, x+lx], [y]*2, y2=[y+ly]*2,
                                                           color=self._projection_props['color'],
                                                           alpha=self._projection_props['alpha'],
                                                           )
            if not isinstance(self._temporal_artist, list):
                self._temporal_artist = [self._temporal_artist]

            self.axes.figure.canvas.draw_idle()

        if event.key == 'r':   # plot radial profile
            rpx, rpy, cxy = pa.radial_profile(self.image, cnt_xy=(event.xdata, event.ydata),
                                              stamp_rad=self._radial_props['stamp_rad'],
                                              recenter=True)
            self.ax_extra.cla()
            self.ax_extra.plot(rpx, rpy, 'x')
            pa.set_plot_props(self.ax_extra, title="radial profile",
                              xlabel="distance to center",
                              ylabel=f'center at ({cxy[0]:.1f}, {cxy[1]:.1f})')
            self.ax_extra.figure.show()
            if self._temporal_artist:
                for t in self._temporal_artist:
                    t.remove()
            self._temporal_artist = self.axes.plot(cxy[0], cxy[1], 'x',
                                                   color=self._radial_props['color'],
                                                   alpha=self._radial_props['alpha'],
                                                   )
            self.axes.figure.canvas.draw_idle()

        if event.key == 'm':   # obtain data coordinates at mouse point
            store_draw_mark(event.xdata, event.ydata, self._mark_props)
        if event.key == 'c':  # obtain data coordinates at mouse point
            stamp_rad = self._centroid_props['stamp_rad']
            yy, xx = pa.subcentroid(self.image, (event.ydata, event.xdata), stamp_rad)
            store_draw_mark(xx, yy, self._centroid_props)
        if event.key == 'd':   # reset marks
            for p in self.axes.patches:
                p.remove()
            self.outs['marks_xy'] = []
            canvas.draw_idle()
        if event.key == 'x':   # invert X
            self.axes.set_xlim(list(self.axes.get_xlim())[::-1])
            self.axes.figure.canvas.draw_idle()
        if event.key == 'y':   # invert Y
            self.axes.set_ylim(list(self.axes.get_ylim())[::-1])
            self.axes.figure.canvas.draw_idle()
        if event.key == 'h':   # project horizontally
            draw_projection(event.xdata, event.ydata,
                            self._projection_props['length'],
                            self._projection_props['width'],
                            self._projection_props['combine_op'], 0)
        if event.key == 'v':   # project horizontally
            draw_projection(event.xdata, event.ydata,
                            self._projection_props['width'],
                            self._projection_props['length'],
                            self._projection_props['combine_op'], 1)
        if event.key == 'q':
            self.disconnect()

        if event.key == '?':
            print(f"'m'ark a new position"
                  f"'c'entroid mark a new position"
                  f"'d'elete the marks"
                  f"'r'adial profile"
                  f"'h'orizontal projection"
                  f"'v'ertical projection"
                  f"invert 'x' axis'"
                  f"invert 'y' axis'"
                  f"'q'uit the interactive mode"
                  )

