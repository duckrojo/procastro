#
# Copyright (C) 2013 Patricio Rojo
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of version 2 of the GNU General
# Public License as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
#

from matplotlib import patches

import procastro as pa
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
import logging
import astropy.timeseries as ts
import astropy.time as apt
from procastro.misc.interactive_graphics import BindingsFunctions
from typing import Optional
import pathlib

TwoValues = tuple[float, float]


__all__ = ['Photometry']


class FilterMessage(logging.Filter):
    def add_needle(self, needle):
        if not hasattr(self, 'needle'):
            # noinspection PyAttributeOutsideInit
            self.needle = []
        self.needle.append(needle)
        return self

    def filter(self, record):
        for needle in self.needle:
            if needle in record.msg:
                return 0
        return 1


tmlogger = logging.getLogger('procastro.timeseries')
for hnd in tmlogger.handlers:
    tmlogger.removeHandler(hnd)
for flt in tmlogger.filters:
    tmlogger.removeFilter(flt)
PROGRESS = 35
handler_console = logging.StreamHandler()
handler_console.setLevel(PROGRESS)
formatter_console = logging.Formatter('%(message)s')
handler_console.setFormatter(formatter_console)
tmlogger.addHandler(handler_console)
tmlogger.propagate = False


def _show_apertures(coords, aperture=None, sky=None,
                    axes=None, sk_color='w', ap_color='w',
                    alpha=0.5, labels=None,
                    clear=True):
    if aperture is None:
        aperture = 10
    if sky is None:
        sky = [15, 20]

    f, ax = pa.figaxes(axes, clear=clear)
    for p in [pp for pp in ax.patches]:
        # noinspection PyArgumentList
        p.remove()
    for t in [tt for tt in ax.texts]:
        # noinspection PyArgumentList
        t.remove()

    if labels is None:
        labels = [''] * len(coords)

    for coo_xy, lab in zip(coords, labels):
        cx, cy = coo_xy

        circ = patches.Circle((cx, cy), sky[1],
                              linewidth=1, edgecolor=sk_color, facecolor=sk_color, alpha=alpha)
        ax.add_patch(circ)

        circ = patches.Circle((cx, cy), aperture,
                              linewidth=1, edgecolor=ap_color, facecolor=ap_color, alpha=alpha)
        ax.add_patch(circ)

        if lab:
            outer_sky = sky[1]
            ax.annotate(lab,
                        xy=(cx, cy + outer_sky),
                        xytext=(cx + 1 * outer_sky,
                                cy + 1.5 * outer_sky),
                        fontsize=20,
                        ha='center')


def _prep_offset(offsets, ignore):
    if ignore is None:
        ignore = []
    if offsets is None:
        offset_list = np.zeros([1, 2])
    else:
        offset_list = np.zeros([max(offsets.keys()) + 1, 2])
        for k, v in offsets.items():
            if k < 0:
                continue
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                offset_list[k, :] = np.array(v)
            elif isinstance(k, int) and v == 0:
                ignore.append(k)
            else:
                raise TypeError(
                    "Unrecognized type for offsets_xy {}. It can be either "
                    "an xy-offset or 0 to indicate skipping".format(v))
    return ignore, offset_list


class _Interactive(BindingsFunctions):

    def __init__(self,
                 stamp_rad: int,
                 labels: list[str],
                 brightest: int = 0,
                 logger: logging.Logger = None,
                 ):

        msg_filter = FilterMessage().add_needle('Using default ')
        # logger.addFilter(msg_filter)
        print("Entering interactive mode:\n"
              "re'c'enter apertures, 'q'uit, toggle 'i'gnore, "
              "keep 'g'oing until drift, <- prev frame, -> next frame, '?' for full help")

        super(_Interactive, self).__init__(None, None, title="Interactive mode for photometry")

        self._prev_brightest_xy: Optional[TwoValues] = None
        self._offset_xy: Optional[TwoValues] = None
        self._ignored: Optional[bool] = None
        self._move: Optional[int] = 0
        self._idx: Optional[int] = None

        self.labels = [''] * len(labels)
        self.labels[brightest] = 'REF'
        self.stamp_rad = stamp_rad
        self._logger = logger
        self._logger_msg_filter = msg_filter

        self.options_reset(config_options=False, quit_option=False)

        self.options_add('left', "Return to previous frame", "_set_move", {'step': -1})
        self.options_add('right', "Advance to next frame", "_set_move", {'step': 1})
        self.options_add('q', "Abort", "_set_move", {'step': None})
        self.options_add('i', "Toggle ignore for this frame", "_add_to_ignore")
        self.options_add('c', 'Recenter the brightest', "_recenter")
        self.options_add('g', 'Keep going until next drift', "_skip_interactive")

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
        self.options_add('h', 'horizontal projection', 'plot_hprojection_exam_2d')
        self.options_add('v', 'vertical projection', 'plot_vprojection_exam_2d')
        self.options_add('x', 'flip X-axis', 'horizontal_flip_2d')
        self.options_add('y', 'flip Y-axis', 'vertical_flip_2d')

    def close(self):
        self._logger.removeFilter(self._logger_msg_filter)
        self.disconnect(close_plot=True)

    # noinspection PyUnusedLocal
    def _skip_interactive(self,
                          xy: Optional[TwoValues],
                          ):
        self._move = 1
        self._skip = True
        self.disconnect(close_plot=False, verbose=False)

    # noinspection PyUnusedLocal
    def _set_move(self,
                  xy: Optional[TwoValues],
                  step: Optional[int],
                  ):
        self._move = step
        self._skip = False
        if step is not None:
            self.disconnect(close_plot=False, verbose=False)

    def new_frame(self, data, filename, idx, ignored, offset_xy, prev_brightest_xy, centers):
        self.clear_data()

        self._move = 0
        self.set_data_2d(data, scale='zscale')
        ax = self.axes_data
        ax.set_xlabel(f'Frame #{idx}: {pathlib.Path(filename).name}')

        self._prev_brightest_xy = prev_brightest_xy
        self._idx = idx
        self._ignored = ignored
        self._offset_xy = offset_xy

        ap_color = sk_color = 'w'
        if ignored:
            ap_color = sk_color = 'r'
        elif offset_xy.sum():
            ap_color = 'b'
        _show_apertures(centers, axes=ax, labels=self.labels,
                        sk_color=sk_color, ap_color=ap_color,
                        clear=False)

        self.connect(verbose=False)

        return self._move, self._skip, self._ignored, self._offset_xy

    # noinspection PyUnusedLocal
    def _add_to_ignore(self, xy):
        self._ignored = not self._ignored
        self._set_move(None, 0)

    def _recenter(self, xy):
        self._offset_xy = (xy[0] - self._prev_brightest_xy[0],
                           xy[1] - self._prev_brightest_xy[1])
        self._set_move(None, 0)

    def __bool__(self):
        return True


class Photometry:
    """
    Applies photometry to a target from the data stored in multiple
    fit frames.

    Fit frames must be stored in a procastro.AstroDir object, the user gives the
    coordinates of the target to be analysed and if possible the positions of
    other stars for reference. The process starts from the first frame with
    the coordinates given and tracks their position as frames are read.

    This process can be controlled by the user by enabling interactive mode
    during initialization. This mode can be used to correct the position of the
    target after a jump or to verify the integrity of the files visually.

    Attributes
    ----------
    surrounding_ap_limit :
       Limit to compute excess flux for a given aperture
    extra_header :
       Name of header item given 'extra'
    extras :
        Values given by 'extra' dict

    Parameters
    ----------
    sci_files : procastro.AstroDir
    target_coords_xy : list
        List containing each target coordinates.
        ([[t1x, t1y], [t2x, t2y], ...])
    offsets_xy : List, optional
    mdark : dict indexed by exposure time, array or AstroFile
        Master dark/bias to be used
    mflat : dict indexed by filter name, array or AstroFile
        Master Flat to be used
    stamp_rad : int, optional
        Radius of the stamp used
    max_skip : int, optional
        Maximum distance allowed between target and stamp center,
        a longer skip will toggle a warning.
    max_counts : int, optional
        Maximum value permitted per pixel
    recenter : bool, optional
        If True, the stamp will readjust its position each frame, following
        the target's movement
    epoch : String, optional
        Header item containing the epoch value
    labels : String list, optional
        Name of each target
    brightest : int, optional
        Index of star to use as position reference
    deg :
    gain : float, optional
        Gain of the telescope
    ron : float, optional
        Read-out-noise
    logfile : String, optional
        If set, logger will save its output on this file
    ignore : List, optional
        Indeces of frames to be ignored
    extra : str, list, dict
        Indicates the extra headers that are to be stored in the output timeseries
    interactive : bool, optional
        Enables interactive mode

    See Also
    --------
    procastro.AstroFile
    procastro.AstroDir
    photometry

    """

    def __init__(self, sci_files, target_coords_xy, offsets_xy=None,
                 mdark=None, mflat=None,
                 stamp_rad=30, outer_ap=1.2,
                 max_skip=8, max_counts=50000, min_counts=4000, max_drift=None,
                 recenter=True,
                 epoch='JD', labels=None, brightest=None,
                 deg=1, gain=None, ron=None,
                 logfile=None, ignore=None, extra=None,
                 logger=None,
                 interactive=False,
                 verbose=True, verbose_procdata=True,
                 ):

        # initialized elsewhere
        self._last_ts = None
        self._skies = None
        self._apertures = None
        self.indexing = None
        self.sci_stamps = None
        self.coords_new_xy = None

        if isinstance(epoch, str):
            self.epochs = sci_files.values(epoch)
        elif hasattr(epoch, '__iter__'):
            self.epochs = epoch
        else:
            raise ValueError(
                "Epoch must be an array of dates in julian date, or a "
                "header's keyword  for the Julian date of the observation")

        self.surrounding_ap_limit = outer_ap

        self.deg = deg
        self.gain = gain
        self.ron = ron
        self.stamp_rad = stamp_rad
        self.max_counts = max_counts
        self.min_counts = min_counts
        self.recenter = recenter
        self.max_skip = max_skip
        if max_drift is None:
            max_drift = stamp_rad/10
        self.max_drift = max_drift
        sci_files = pa.AstroDir(sci_files, mbias=mdark, mflat=mflat)
        self._astrodir = sci_files

        ignore, offset_list = _prep_offset(offsets_xy, ignore)

        if logger is not None:
            self._logger = logger
        else:
            if logfile is None:
                tmlogger.propagate = False
                self._logger = tmlogger
            else:
                tmlogger.propagate = False

                # use logger instance that includes filename to allow different
                # instance of photometry with different loggers as long as they
                # use different files
                self._logger = logging.getLogger(
                    'procastro.timeseries.{}'
                    .format(os.path.basename(logfile)
                            .replace('.', '_')))
                self._logger.setLevel(logging.INFO)
                # in case of using same file name start new with loggers
                for hnd_tmp in self._logger.handlers:
                    self._logger.removeHandler(hnd_tmp)

                handler = logging.FileHandler(logfile, 'w')
                formatter = logging.Formatter("%(asctime)s: %(name)s "
                                              "%(levelname)s: %(message)s")
                handler.setFormatter(formatter)
                handler.setLevel(logging.INFO)
                self._logger.addHandler(handler)

                print("Detailed logging redirected to {}".format(logfile))
        newline = '\n  '
        self._logger.info(f"procastro.timeseries.Photometry execution on:\n"
                          f"  {newline.join([f.filename for f in sci_files])}")

        # label list
        if isinstance(target_coords_xy, dict):
            coords_user_xy = list(target_coords_xy.values())
            labels = list(target_coords_xy.keys())
        elif isinstance(target_coords_xy, (list, tuple)):
            coords_user_xy = list(target_coords_xy)
        else:
            raise TypeError(f"target_coords_xy ({target_coords_xy}) type is invalid")

        try:
            if labels is None:
                labels = []
            nstars = len(target_coords_xy)
            if len(labels) > nstars:
                labels = labels[:nstars]
            elif len(labels) < nstars:
                labels = list(labels) + list(np.arange(len(labels),
                                                       nstars).astype(str))
        # TODO: Define this exception type
        except (TypeError, ValueError):
            raise ValueError("Coordinates of target stars need to be "
                             "specified as dictionary or as a list of 2 "
                             "elements, not: {}"
                             .format(str(target_coords_xy), ))
        self.labels = labels
        self.coords_user_xy = coords_user_xy

        # The following is to search for the brightest star...
        # rough aperture photometry performed
        if brightest is None:
            flxs = []
            data = sci_files[0].reader(verbose=verbose_procdata)
            for trgx, trgy in coords_user_xy:
                if not (0 < trgy < data.shape[0]) and not (0 < trgx < data.shape[1]):
                    raise ValueError(
                        f"Given coordinates ({trgx}, {trgy}) is beyond data size ({data.shape[1]}, {data.shape[0]})")
                stamp = pa.subarray(data,
                                    pa.subcentroid(data,
                                                   (trgy, trgx),
                                                   stamp_rad),
                                    stamp_rad)
                d = pa.radial(stamp, (stamp_rad, stamp_rad))
                sky_val = np.median(stamp[(d < stamp_rad) * (d > stamp_rad-5)])
                flxs.append((stamp[d < stamp_rad/2] - sky_val).sum())
            brightest = np.argmax(flxs)
        self.brightest = brightest

        self._logger.info(f" Initial guess received for {len(target_coords_xy)} targets, "
                          f"reference brightest '{self.labels[brightest]}'.")
        self._logger.info(
            "Initial coordinates {}"
            .format(", ".join([f"{lab} {coo}"
                               for lab, coo in zip(labels, coords_user_xy)])
                    ))

        self._get_stamps(ignore,
                         offset_list,
                         interactive=interactive,
                         verbose=verbose,
                         verbose_procdata=verbose_procdata,
                         )

        if extra is None:
            self.extra_header = {}
            self.extras = {}
        else:
            if isinstance(extra, list):
                self.extra_header = {k: k for k in extra}
            elif isinstance(extra, dict):
                self.extra_header = extra.copy()
            elif isinstance(extra, str):
                self.extra_header = {extra: extra}
            else:
                raise TypeError(f"Header specification 'extra' ({extra}) was not understood. "
                                f"It can be str, list, or dictionary (fits_value: column_value) ")
            # Storing extras and frame_id with the original indexing.
            self.extras = {x: list(v)
                           for x, v in zip(extra, zip(*sci_files.values(*self.extra_header.keys(),
                                                                        single_in_list=True)))}

        # storing indexing only for those not ignored

    def _get_stamps(self, ignore, offsets_xy,
                    interactive=False,
                    verbose=True, verbose_procdata=True,
                    ):
        """
        ...

        Parameters
        ----------
        offsets_xy : List or numpy.ndarray, optional

        ignore : int list, optional
            List with the indexes of frames to be ignored
        interactive : bool, optional
            Enables interactive mode. See Notes below.

        Notes
        -----
        Interactive Mode:
        Enbling this setting gives manual control over the reduction process,
        the following commands are available:
            * Left and Right keys : Change to previous/next frame
            * c : Recenter stamps based on mouse position
            * i : De/ignore current frame
            * g : Keep going until a major drift occurs
            * q : Exit interactive mode and stop the process

        Frames can be flagged by pressing 1-9

        Returns
        -------

        """

        if ignore is None:
            ignore = []

        max_change_allowed = self.max_drift
        n_targets = len(self.coords_user_xy)
        stamp_rad = self.stamp_rad
        sci_files = self._astrodir
        max_skip = self.max_skip
        recenter = self.recenter
        logger = self._logger
        labels = self.labels

        n_files = len(sci_files)

        tmp = np.zeros([n_files, 2])
        if offsets_xy is not None:
            tmp[:len(offsets_xy), :] = offsets_xy
        offsets_xy = tmp

        all_centers_xy = np.zeros([n_targets, n_files, 2])
        all_cubes = np.zeros([n_targets, n_files, stamp_rad * 2 + 1, stamp_rad * 2 + 1])
        indexing = [-1] * n_files

        if verbose:
            logger.info(" Obtaining stamps for {} files: ".format(n_files))

        if interactive:
            interactive = _Interactive(stamp_rad, labels, logger=logger, brightest=self.brightest)
            self.inter = interactive

        prev_centers_xy = [(xx, yy) for xx, yy in self.coords_user_xy]
        previous_distance = None
        skip_interactive = False
        to_store = 0
        idx = 0
        step = 1

        while idx < n_files:
            long_skip = False
            filename = sci_files[idx].filename
            off = offsets_xy[idx]

            offseted = np.abs(off).sum() > 0
            ignored = idx in ignore
            if offseted:
                pass

            if step != 0:
                try:
                    reduced_data = sci_files[idx].reader(verbose=verbose_procdata)
                except OSError:
                    logger.warning(f"Corrupt file {sci_files[idx].filename}, forcibly added to the ignore list")
                    if not ignored:
                        ignore.append(idx)
                    idx += 1
                    continue

            centers_xy = []
            cubes = []

            for label, prev_center_xy in zip(labels, prev_centers_xy):
                cx, cy = prev_center_xy[0] + off[0], prev_center_xy[1] + off[1]
                if recenter:
                    ncy, ncx = pa.subcentroid(reduced_data, [cy, cx], stamp_rad)
                else:
                    ncy, ncx = cy, cx
                if ncy < 0 or ncx < 0 or ncy > reduced_data.shape[0] or ncx > reduced_data.shape[1]:
                    raise ValueError(f"Centroid for frame #{idx}({filename}) falls outside data"
                                     f" for target {label}. Initial/final center was:"
                                     f" [{cx:.2f}, {cy:.2f}]/[{ncx:.2f}, {ncy:.2f}] Offset: {off}")
                cubes.append(pa.subarray(reduced_data, [ncy, ncx], stamp_rad, padding=True))
                centers_xy.append((ncx, ncy))

                skip = np.sqrt((cx - ncx) ** 2 + (cy - ncy) ** 2)
                if skip > max_skip and not long_skip:
                    long_skip = True
                    # logger.warning(f"Large jump ({cx}, {cy}) -> ({ncx}, {ncy}). Stamprad: {stamp_rad} pix")
                    logger.warning(f"Large jump of {skip:.1f} pixels has occurred "
                                   f"on frame #{idx}{'(User Coordinates!)' if not idx else ''}: {filename}")

            centers_xy = np.array(centers_xy)
            cubes = np.array(cubes)

            drifting = False
            distance_to_brightest_vector = centers_xy - centers_xy[self.brightest][None, :]
            distance_to_brightest = np.sqrt((distance_to_brightest_vector ** 2).sum(1))
            if previous_distance is not None:
                change = np.absolute(distance_to_brightest - previous_distance)
                max_change = max(change)
                if max_change > max_change_allowed and not ignored:
                    drifting = True
                    logger.warning(f"Found star drifting from brightest by {max_change:.1f} pixels "
                                   f"between consecutive frames for target {labels[np.argmax(max_change)]}."
                                   )

            step = 1
            if interactive and (not skip_interactive or drifting):
                step, skip_interactive, ignored, offset = interactive.new_frame(reduced_data, filename,
                                                                                idx, ignored, off,
                                                                                prev_centers_xy[self.brightest],
                                                                                centers_xy)
                offsets_xy[idx] = offset
                if ignored and idx not in ignore:
                    ignore.append(idx)
                elif not ignored and idx in ignore:
                    ignore.remove(idx)

            if not step:
                continue
            elif step < 0:
                # reduce to_store only if it was stored in that previous frame.
                to_store += step * (indexing[to_store + step] == idx + step)
                idx += step
                continue

            indexing[to_store] = idx
            all_cubes[:, to_store, :, :] = cubes
            all_centers_xy[:, to_store, :] = centers_xy

            msg = ''
            if idx in ignore:  # skipped at least 1
                msg += 'S'
            if offseted:  # drift applied
                msg += "O"
            if long_skip:  # jump of a target
                msg += "J"
            if drifting:  # drift from brightest
                msg += "D"
            symbol = '=' if idx % 100 == 99 else (':' if idx % 10 == 9 else '.')
            print(f'{msg}{symbol}', end='', flush=True)

            prev_centers_xy = centers_xy
            previous_distance = distance_to_brightest
            to_store += 1
            idx += step

        print('')
        if verbose:
            logger.info("Skipped {} flagged frames".format(n_files - to_store))

        offsets_out_xy = {}
        for idx in range(len(offsets_xy)):
            if idx in ignore:
                offsets_out_xy[idx] = 0
            elif offsets_xy[idx].sum() != 0:
                offsets_out_xy[idx] = tuple(offsets_xy[idx])

        dict_union = ',\n              '
        if interactive:
            if verbose:
                off = {k: f"[{v[0]:.2f}, {v[1]:.2f}]" if isinstance(v, tuple)
                       else v for k, v in offsets_out_xy.items()}
                logger.info("This interactive run was equivalent to use the keyword:")
                logger.info("offsets_xy = {\n              "
                            f"{dict_union.join([f'{k}: {v}' for k, v in off.items()])}"
                            ",\n              }")

            interactive.disconnect(close_plot=True, verbose=False)

        self.sci_stamps = all_cubes[:, :to_store, :, :]
        self.indexing = indexing[:to_store]
        self.coords_new_xy = all_centers_xy[:, :to_store, :]

    def set_max_counts(self, counts):
        self.max_counts = counts

    def set_min_counts(self, counts):
        self.min_counts = counts

    def photometry(self,
                   aperture, sky=None, skies=None,
                   deg=None, max_counts=None, min_counts=None,
                   outer_ap=None, verbose=True, progress_indicator=None,
                   ) -> ts.TimeSeries:
        """
        Verifies parameters given and applies photometry through cpu_phot

        Parameters
        ----------
        verbose
        progress_indicator
        min_counts
        skies
        aperture :
        sky :
        deg :
        max_counts : int, optional
            Maximum value allowed per pixel, raises a warning if a target does
        outer_ap :
            Outer ring as a fraction of aperture, to report surrounding region
        """
        # make sure that aperture and sky values are compatible.

        if skies is None:
            if sky is None:
                raise ValueError("Both sky and skies for .photometry() cannot be None")
            skies = [sky]

        if isinstance(aperture, (int, float)):
            self._apertures = [aperture]
        elif isinstance(aperture, (list, tuple, np.ndarray)):
            self._apertures = aperture

        max_aperture = np.max(aperture)
        skies = [(sky1 if sky1 > max_aperture else max_aperture+2, sky2) for sky1, sky2 in skies]
        self._skies = skies

        if deg is not None:
            self.deg = deg
        if max_counts is not None:
            self.set_max_counts(self.max_counts)
        if min_counts is not None:
            self.set_min_counts(self.min_counts)
        if outer_ap is not None:
            self.surrounding_ap_limit = outer_ap

        if self._apertures is None or self._skies is None:
            raise ValueError(
                "ERROR: aperture photometry parameters are incomplete. Either "
                "aperture photometry radius or sky annulus were not giving. "
                "Please call photometry with the following keywords: "
                "photometry(aperture=a, sky=s) or define aperture "
                "and sky when initializing Photometry object.")

        if any([self._apertures[i] > self.stamp_rad
                for i in range(len(self._apertures))]):
            raise ValueError(
                "Aperture photometry ({}) shouldn't be higher than radius of "
                "stamps ({})".format(self._apertures,
                                     self.stamp_rad))
        if np.min(np.array(self._skies), 0)[0] < np.max(self._apertures):
            raise ValueError(
                "Aperture photometry ({}) shouldn't be higher than inner sky "
                "radius ({})".format(self._apertures,
                                     ", ".join([s[0] for s in self._skies])))

        if self.stamp_rad < np.max(np.array(self._skies), 0)[1]:
            raise ValueError(
                "External radius of sky ({}) shouldn't be higher than stamp "
                "radius ({})".format(", ".join([s[1] for s in self._skies]),
                                     self.stamp_rad))

        ts_return = self.cpu_phot(verbose=verbose, progress_indicator=progress_indicator)
        
        self._last_ts = ts_return
        return ts_return

    def remove_from(self, idx):
        """
        Removes file from index 'idx' owwards, from the list of frames to be processed

        idx : int
        """
        if not isinstance(idx, int):
            raise TypeError("idx can only be indexing")
        idx_skipping = self.indexing.index(idx)
        self.indexing = self.indexing[:idx_skipping]
        self.sci_stamps = self.sci_stamps[:, :idx_skipping, :, :]
        self.coords_new_xy = self.coords_new_xy[:, :idx_skipping, :]
        # self.frame_id = self.frame_id[:idx]
        for v in self.extras.keys():
            self.extras[v] = self.extras[v][:idx]
        self._astrodir = self._astrodir[:idx]

    def cpu_phot(self, verbose=True, progress_indicator=None):
        """
        Calculates the CPU photometry for all frames
        #TODO improve this docstring

        Returns
        -------
        procastro.TimeSeries :
            TimeSeries object containing the resulting data

        Notes
        -----
        This method requires scipy to work properly

        See Also
        --------
        procastro.TimeSeries : Object used to store the output

        """

        if progress_indicator is None:
            progress_indicator = 'X' if verbose else ''
        if isinstance(self._apertures, (list, tuple, np.ndarray)):
            aperture = self._apertures
        else:
            aperture = [self._apertures]

        nt, ns, ny, ny = self.sci_stamps.shape

        skies = self._skies

        data_store = {}

        if verbose:
            print("Processing CPU photometry for {0} targets: "
                  .format(len(self.sci_stamps)), end='')
            sys.stdout.flush()
        ref_labels = [self.labels[0]] + [f'ref{i:d}' for i in range(1, nt)]
        for label, ref_label, target, centers_xy, target_idx in zip(self.labels,
                                                                    ref_labels,
                                                                    self.sci_stamps,
                                                                    self.coords_new_xy,
                                                                    range(nt)):  # For each target

            # n_epoch, 2
            center_stamp_xy = np.array(centers_xy) % 1 + np.array([self.stamp_rad, self.stamp_rad])[None, :]
            xx, yy = np.mgrid[0:target.shape[2], 0: target.shape[1]]
            dx = xx[None, :, :] - center_stamp_xy[:, 0][:, None, None]
            dy = yy[None, :, :] - center_stamp_xy[:, 1][:, None, None]
            d = np.sqrt(dx*dx + dy*dy)

            data_store[f"centerx_{ref_label}"] = np.array(centers_xy)[:, 0]
            data_store[f"centery_{ref_label}"] = np.array(centers_xy)[:, 1]

            peak = 0
            for sky in skies:
                if sky[0] >= sky[1]:
                    continue
                masked_sky = np.ma.array(target, mask=(d < sky[0])+(d > sky[1]))
                sky_std = masked_sky.std(axis=(1, 2))
                sky_avg = np.ma.median(masked_sky, axis=(1, 2))
                n_pix_sky = np.ma.count_masked(masked_sky, axis=(1, 2))

                for ap in aperture:
                    apsky_label = f"ap{ap:.1f}_ski{sky[0]:.1f}_sko{sky[1]:.1f}"

                    skyless = target - sky_avg[:, None, None]
                    ap_mask = d > ap
                    masked_ap = np.ma.array(skyless, mask=ap_mask)
                    flux = masked_ap.sum(axis=(1, 2))
                    n_pix_ap = np.ma.count_masked(masked_ap, axis=(1, 2))

                    var_flux = flux / self.gain
                    var_sky = sky_std ** 2 * n_pix_ap * (1 + n_pix_ap.astype(float)) / n_pix_sky
                    var_total = var_sky + var_flux + self.ron * self.ron * n_pix_ap
                    error = np.sqrt(var_total)

                    excess = np.ma.array(skyless,
                                         mask=(ap > d) + (d > self.surrounding_ap_limit * ap)).sum(axis=(1, 2))
                    peak = masked_ap.max(axis=(1, 2))

                    if ap > sky[0]:
                        continue

                    data_store[f'flux_{ref_label}_{apsky_label}'] = flux
                    data_store[f'err_flux_{ref_label}_{apsky_label}'] = error
                    data_store[f'peak_{ref_label}_{apsky_label}'] = peak
                    data_store[f'excess_{ref_label}_{apsky_label}'] = excess

                    mom2 = (masked_ap*(masked_ap > 0) * d*d).sum(axis=(1, 2))

                    skew_x = (masked_ap*(masked_ap > 0) * (dx ** 3)).sum(axis=(1, 2))
                    skew_y = (masked_ap*(masked_ap > 0) * (dy ** 3)).sum(axis=(1, 2))
                    mom3m = np.sqrt(skew_x * skew_x + skew_y * skew_y)
                    mom3a = np.arctan2(skew_y, skew_x)

                    kurt = (masked_ap*(masked_ap > 0) * (d ** 4)).sum(axis=(1, 2))

                    data_store[f'mom2_{ref_label}_{apsky_label}'] = mom2
                    data_store[f'mom3m_{ref_label}_{apsky_label}'] = mom3m
                    data_store[f'mom3a_{ref_label}_{apsky_label}'] = mom3a
                    data_store[f'mom4_{ref_label}_{apsky_label}'] = kurt

            too_much_mask = peak > self.max_counts
            too_little_mask = peak < self.min_counts
            for mask, lab, thresh in ((too_much_mask, "ABOVE", self.max_counts),
                                      (too_little_mask, "BELOW", self.min_counts)):
                n_mask = mask.sum()
                if n_mask:
                    self._logger.warning(f"Object {label}"
                                         f" ha{'ve' if n_mask > 1 else 's'}"
                                         f" counts {lab} the threshold ({thresh})"
                                         f" on frame{'s' if n_mask > 1 else ''}"
                                         f" {', '.join(np.array(self.indexing)[mask].astype(str))} "
                                         )

            # for data, center_xy, non_ignore_idx, epochs_idx \
            #         in zip(target, centers_xy, self.indexing, range(ns)):
            #
            #
            #         # Following to compute FWHM by fitting gaussian
            #         res2 = res[d < sky[1]].ravel()
            #         d2 = d[d < sky[1]].ravel()
            #         to_fit = lambda dd, h, sigma: h * dp.gauss(dd, sigma, ndim=1)
            #         try:
            #             sig, cov = op.curve_fit(to_fit,
            #                                     d2,
            #                                     res2,
            #                                     sigma=1 / np.sqrt(np.absolute(res2)),
            #                                     p0=[max(res2), 3])
            #         except RuntimeError:
            #             sig = np.array([0, 0, 0])
            #         fwhm_g = 2.355 * sig[1]
            #
            #         fwhm_gs.append(fwhm_g)
            #
            if progress_indicator:
                print(progress_indicator, end='', flush=True)
                sys.stdout.flush()

        data_store |= {self.extra_header[k]: [self.extras[k][i] for i in self.indexing]
                       for k in self.extra_header.keys()}
        data_store |= {"indexing": self.indexing}

        if verbose:
            print('')

        return ts.TimeSeries(time=apt.Time([self.epochs[i] for i in self.indexing], format='jd'),
                             data=data_store)
    
    def infos_available(self):
        ret = []
        for a in self._last_ts.colnames:
            if a == 'time':
                continue
            fields = a.split("_", 3)
            field = fields[0]
            if field == "err":
                field += f"_{fields[1]}"
            ret.append(field)

        return set(ret)

    def last_coordinates(self, pos=None):
        """
        Returns a dictionary with each target's position from the last frame.
        Useful if continued on a separate object

        Returns
        -------
        dict :
            Dictionary contianing each targets coordinates
        """
        if pos is None:
            ret_idx = -1
        else:
            ret_idx = self.indexing.index(pos)
        return {self.labels[k]: list(self.coords_new_xy[k, ret_idx, :].astype(int))
                for k in range(len(self.labels))}

    def plot_radialprofile(self, targets=None, axes=1, frame=0,
                           recenter=True, clear=True, **kwargs):
        """
        Plots a Radial Profile from the data using procastro.radialprofile

        Parameters
        ----------
        targets: int/str
            Target specification for re-centering. Either an integer for
            specific target.
        axes : int, plt.Figure, plt.Axes
        frame : int, optional
        recenter : bool, optional
            If True, targets will be tracked as they move
        clear : bool, True
        """

        colors = ['kx', 'rx', 'bx', 'gx', 'k^', 'r^', 'b^', 'g^', 'ko', 'ro',
                  'bo', 'go']
        fig, ax = pa.figaxes(axes, clear=clear)

        ax.cla()
        ax.set_title('Radial profile')
        ax.set_xlabel('Distance (in pixels)')
        ax.set_ylabel('ADU')
        if targets is None:
            targets = self.labels
        elif isinstance(targets, str):
            targets = [targets]
        elif isinstance(targets, (list, tuple)) and isinstance(targets[0], (int,)):
            targets = [self.labels[a] for a in targets]
        elif isinstance(targets, int):
            targets = [self.labels[targets]]

        stamp_rad = self.stamp_rad

        for stamp, coords_xy, color, lab in zip(self.sci_stamps, self.coords_new_xy,
                                                colors, self.labels):
            if lab in targets:
                cx, cy = coords_xy[frame]
                distance, value, center = \
                    pa.radial_profile(stamp[frame],
                                      [stamp_rad + cx % 1, stamp_rad + cy % 1],
                                      stamp_rad=stamp_rad,
                                      recenter=recenter)
                ax.plot(distance,
                        value,
                        color,
                        label="{0:s}: ({1:.1f}, {2:.1f}) -> ({3:.1f}, {4:.1f})"
                        .format(lab,
                                coords_xy[frame][0],
                                coords_xy[frame][1],
                                coords_xy[frame][0] - stamp_rad + center[0],
                                coords_xy[frame][1] - stamp_rad + center[1]),
                        )

        pa.set_plot_props(ax, **kwargs)

    def showstamp(self, target=None, stamp_rad=None, axes=None,
                  first=0, last=-1, n_show=None, ncol=None, annotate=True,
                  imshow=None, clear=True, **kwargs):
        """
        Show the star at the same position for the different frames

        Parameters
        ----------
        target : None for the first key()
        stamp_rad : int, optional
            Stamp radius
        axes : int, matplotlib.pyplot Axes, optional
        first : int, optional
            First frame to show
        last : int, optional
            Last frame to show. -1 will show all stamps
        n_show : int, optional
            Indicates the number of figures to present. It overwrites the
            value of last
        ncol : int, optional
            Number of columns
        annotate : bool, optional
            If True, it will include the frame number with each stamp
        imshow : matplotlib.axes.Axes, matplotlib.figure.Figure
            Axes or figure where to examine a stamp
        clear : bool, optional
        """
        if target is None:
            target = 0
        elif isinstance(target, str):
            target = self.labels.index(target)

        if n_show is not None:
            last = first + n_show

        # change first and last to skipped indexing
        first = list(np.array(self.indexing) >= first).index(True)
        if last < 0:
            last += len(self._astrodir)
        try:
            last = list(np.array(self.indexing) >= last).index(True)
        except ValueError:
            last = len(self.indexing) - 1
        n_images = last - first + 1

        if stamp_rad is None or stamp_rad > self.stamp_rad:
            stamp_rad = self.stamp_rad

        if ncol is None:
            ncol = int(np.sqrt(n_images) * 1.3)
        nrow = int(np.ceil(float(n_images) / ncol))

        stamp_d = 2 * stamp_rad + 1
        array = np.zeros([nrow * (stamp_d + 2), ncol * (stamp_rad * 2 + 3)])
        for data, idx in zip(self.sci_stamps[target][first:last + 1],
                             range(first, last + 1)):
            pos_idx = idx - first
            xpos = 1 + (pos_idx % ncol) * (stamp_d + 2)
            ypos = 1 + (pos_idx // ncol) * (stamp_d + 2)
            array[ypos:ypos + stamp_d, xpos: xpos + stamp_d] = data

        f_stamp, ax_stamp = pa.figaxes(axes, clear=clear)
        pa.imshowz(array, axes=ax_stamp, show=False)
        if annotate:
            for idx in range(first, last + 1):
                pos_idx = idx - first
                xpos = 1 + stamp_rad / 5 + (pos_idx % ncol) * (stamp_d + 2)
                ypos = 1 + stamp_rad / 10 + (pos_idx // ncol) * (stamp_d + 2)
                plt.text(xpos, ypos, self.indexing[idx])

        if imshow is not None:
            def onclick(event):
                if event.inaxes != ax_stamp:
                    return
                xx, yy = event.xdata, event.ydata
                goto_idx = self.indexing[first:last][int(ncol * (yy // (stamp_d + 2)) + xx // (stamp_d + 2))]
                ax_show.cla()
                f_show.show()
                self.imshowz(goto_idx, axes=ax_show)

            f_show, ax_show = pa.figaxes(imshow)

            # noinspection PyUnusedLocal
            dummy = ax_stamp.figure.canvas.mpl_connect('button_press_event',
                                                       onclick)
        pa.set_plot_props(ax_stamp, **kwargs)

    def plot_drift(self, target=None, axes=None, **kwargs):
        """
        Plots a target movement between frames

        Parameters
        ----------
        target : str, optional
            Name of the target
        axes : int, matplotlib.pyplot Axes, optional
        """
        colors = ['k-', 'r-', 'b-', 'g-', 'k--', 'r--', 'b--', 'g--', 'k:',
                  'r:', 'b:', 'g:']
        fig, ax = pa.figaxes(axes)

        if target is None:
            labels = self.labels
            coords_xy = self.coords_new_xy
        elif isinstance(target, int):
            labels = [self.labels[target]]
            coords_xy = [self.coords_new_xy[target]]
        elif isinstance(target, str):
            labels = [target]
            coords_xy = [self.coords_new_xy[self.labels.index(target)]]
        else:
            raise TypeError("target type not identified")

        for label, coord_xy, color in zip(labels, coords_xy, colors):
            xx, yy = coord_xy.transpose()
            ax.plot(xx - xx[0], yy - yy[0], color, label=label)

        legend_dict = {'bbox_to_anchor': (0., 1.02, 1., .302), 'loc': 3,
                       'ncol': int(len(labels) // 2),
                       'mode': "expand",
                       'borderaxespad': 0.,
                       'prop': {'size': 8}}
        pa.set_plot_props(ax, legend=legend_dict, **kwargs)

    def plot_extra(self, x_id=None, axes=None):
        """

        Parameters
        ----------
        x_id:
        axes:
        """
        fig, ax, x = pa.figaxes_xdate(self.epochs, axes=axes)

        ax.plot(x, self.extras[x_id])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(x_id)

    def imshowz(self, frame=0,
                ap_color='w', sk_color='LightCyan',
                alpha=0.6, axes=None, reference=None,
                annotate=True, cnt=None, interactive=True,
                clear=True, show=False, **kwargs):
        """
        Plots the image from the fits file, after being processed using the
        zscale Algorithm

        Parameters
        ----------
        show: bool
           Whether to .show() the plot
        frame:
        ap_color: string, optional
        sk_color: string, optional
        alpha : float, optional
        axes : int, matplotlib.pyplot Axes, optional
        reference:
        annotate: bool, optional
            Displays target's name
        cnt: tuple, string, optional
            It can be an XY tuple or a str to identify a specific target
            position at the required frame
        interactive: bool, optional
            Enables interactive mode
        clear: bool, optional
            Clear previous axes content
        """
        f, ax = pa.figaxes(axes, clear=clear)
        ax.cla()
        d = self._astrodir[frame]

        pa.imshowz(d, axes=ax,
                   show=False, **kwargs)

        if reference is None:
            reference = frame
        elif reference < 0:
            reference += frame
        reference_ignore = self.indexing.index(reference)

        def coords_n_cnt(ref, trg):
            coord = list(zip(*self.coords_new_xy))[ref]
            if trg is None:
                cnt_label = 'origin'
                cnt_coord_xy = [0, 0]
            elif isinstance(trg, int):
                cnt_label = self.labels[trg]
                cnt_coord_xy = coord[trg]
            elif isinstance(trg, str):
                cnt_label = trg
                cnt_coord_xy = coord[self.labels.index(trg)]
            else:
                raise TypeError("cnt has to be label identification")
            return coord, cnt_label, cnt_coord_xy

        coords, ref_label, ref_xy = coords_n_cnt(reference_ignore, cnt)

        # noinspection PyDefaultArgument
        def _onkey(event, store=[], ref_input=[]):
            # Handles keyboard input
            if event.inaxes != ax:
                return
            xx, yy = event.xdata, event.ydata
            ref = reference_ignore
            cnt_tmp = cnt
            if len(store) == 0:
                ref_input.append(False)
                store.extend([ref_label, ref_xy])
            if ref_input[0]:  # Having to grab one number at a time
                if len(event.key) == 1 and (ord('0') <= ord(event.key) <= ord('9')):
                    for i in range(len(ref_input)):
                        if ref_input[i] is False:
                            ref_input[i] = event.key
                            break
                    else:
                        ref_input.append(event.key)
                    event.inaxes.set_xlabel("Input Reference: {}"
                                            .format("".join([a for a
                                                             in ref_input
                                                             if a is not True]
                                                            )
                                                    )
                                            )
                    event.inaxes.figure.show()
                    return
                elif event.key == 'enter':
                    ref = self.indexing.index(int("".join(ref_input[1:])))
                    while ref_input.pop() is not True:
                        pass
                    ref_input.append(False)
                    event.inaxes.set_xlabel("")
                else:
                    return
                event.inaxes.figure.show()
            elif event.key == 'c':
                dist = np.sqrt(((np.array(coords) - np.array([xx, yy])[None, :]) ** 2).sum(1))
                cnt_tmp = self.labels[np.argmin(dist)]
            elif event.key == 'o':
                print(store)
                print("Offset to {}: {}, {}"
                      .format(store[0], xx - store[1][0], yy - store[1][1]))
                return
            elif event.key == 'r':
                ref_input[0] = True
                event.inaxes.set_xlabel("Input Reference: ")
                event.inaxes.figure.show()
                return
            else:
                return
            n_coords, n_ref_label, n_ref_xy = coords_n_cnt(ref, cnt_tmp)
            # noinspection PyUnusedLocal
            store[:] = [n_ref_label, n_ref_xy]
            _show_apertures(coords, aperture=self._apertures[0], sky=self._skies[0],
                            axes=ax, labels=annotate and self.labels or None,
                            sk_color=sk_color, ap_color=ap_color, alpha=alpha)
            ax.set_ylabel("Frame #{}{}"
                          .format(frame, reference != frame
                                  and ", apertures from #{}".format(reference)
                                  or ""))

        def _onclick(event):
            # Handles mouse input
            if event.inaxes != ax:
                return
            xx, yy = event.xdata - ref_xy[0], event.ydata - ref_xy[1]
            print("\nFrame #{} (X, Y, Flux) = ({:.1f}, {:.1f}{})]"
                  .format(frame,
                          event.xdata, event.ydata,
                          d[event.ydata, event.xdata]))
            print("Distance to '{}'{}: (x,y,r) = ({:.1f}, {:.1f}, {:.1f})"
                  .format(ref_label,
                          reference != frame
                          and " on frame #{}".format(reference)
                          or "",
                          xx, yy,
                          np.sqrt(xx * xx + yy * yy),
                          ))

        if interactive:
            f.canvas.mpl_connect('button_press_event', _onclick)
            f.canvas.mpl_connect('key_press_event', _onkey)

        _show_apertures(coords, aperture=self._apertures[0], sky=self._skies[0],
                        axes=ax, labels=annotate and self.labels or None,
                        sk_color=sk_color, ap_color=ap_color, alpha=alpha,
                        clear=False)
        ax.set_ylabel("Frame #{}{}".format(frame, reference != frame
                                           and f", apertures from #{reference}"
                                           or ""))

        plt.tight_layout()
        pa.set_plot_props(ax, show=show, **kwargs)
