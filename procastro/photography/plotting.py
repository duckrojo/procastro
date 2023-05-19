#
# Copyright (C) 2021 Patricio Rojo
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

__all__ = ['PhotoPlotting']

import pandas

import procastro as pa
import astropy.units as u
import numpy as np
import astropy.time as apt


class PhotoPlotting:
    def __init__(self, umbra=40):
        self._df = None  # type: pandas.DataFrame
        self._figs = []
        self.umbra = umbra
        self._ref_time = None
        self._dataset_name = None

    def set_ref_time(self, ref_time=None, ref_frame=0):
        df = self._df
        if ref_time is None:
            ref_time = self._ref_time
        if ref_time is None:
            ref_time = df['date'][ref_frame]
        if isinstance(ref_time, int):
            ref_time = df['date'][0] + ref_time*u.s
        if isinstance(ref_time, str):
            ref_time = apt.Time(ref_time)

        df['delta_time'] = [(dt - ref_time).to(u.s).value for dt in df['date']]
        df['delta_post'] = df['delta_time'] + df['exposure']
        self._ref_time = ref_time

    def plot_ev(self, ref_frame=0, ref_time=None,
                ylabel="Ev",
                marker='.', ls='', color='red',
                label=None, legend=False, umbra=True, save=None,
                xlims=None, ax=None, overwrite=False):
        self.set_ref_time(ref_time, ref_frame)
        df = self._df
        f, ax = pa.figaxes(ax, overwrite=overwrite)
        if f not in self._figs:
            self._figs.append(f)

        ax.plot(df['delta_time'], df['ev'], label=label,
                marker=marker, ls=ls, color=color)
        ax.set_title(self._dataset_name)
        ax.set_ylabel(ylabel)
        ax.yaxis.label.set_color(color)

        if umbra:
            ax.axvspan(0, self.umbra, color="gray", alpha=0.6)
        ax.set_xlim(xlims)
        f.tight_layout()
        if legend:
            ax.legend()
        if save is not None:
            f.savefig(save)

    def plot_aperture(self, ref_frame=0, ref_time=None,
                marker='.', ls='', color='red',
                label=None, legend=False,
                xlims=None, ax=None, overwrite=False):

        self.set_ref_time(ref_time, ref_frame)
        df = self._df
        f, ax = pa.figaxes(ax, overwrite=overwrite)
        if f not in self._figs:
            self._figs.append(f)

        ax.plot(df['delta_time'], df['fnumber'], label=label,
                marker=marker, ls=ls, color=color)
        ax.set_title(self._dataset_name)
        ax.set_ylabel("fnumber")
        ax.yaxis.label.set_color(color)

        ax.set_xlim(xlims)
        f.tight_layout()
        if legend:
            ax.legend()

    def plot_iso(self, ref_frame=0, ref_time=None,
                marker='.', ls='', color='red',
                label=None, legend=False,
                xlims=None, ax=None, overwrite=False):

        self.set_ref_time(ref_time, ref_frame)
        df = self._df
        f, ax = pa.figaxes(ax, overwrite=overwrite)
        if f not in self._figs:
            self._figs.append(f)

        ax.plot(df['delta_time'], df['iso'], label=label,
                marker=marker, ls=ls, color=color)
        ax.set_title(self._dataset_name)
        ax.set_ylabel("iso")
        ax.yaxis.label.set_color(color)

        ax.set_xlim(xlims)
        f.tight_layout()
        if legend:
            ax.legend()

    def plot_exposure(self, ref_time=None, ref_frame=0,
                      marker='|', ls='-', color='blue',
                      label=None, legend=False,
                      xlims=None, ax=None,
                      overwrite=False):
        self.set_ref_time(ref_time, ref_frame)
        df = self._df
        f, ax = pa.figaxes(ax, overwrite=overwrite)
        if f not in self._figs:
            self._figs.append(f)

        nans = np.empty(len(df['delta_time']))
        nans[:] = np.NaN
        delta_time_3 = [item for sublist in zip(df['delta_time'], df['delta_post'], nans) for item in sublist]
        exposure_sec_3 = [item for item in df['exposure'] for _ in (1, 2, 3)]

        ax.semilogy(df['delta_time'], df['exposure'], label=label,
                    marker=marker, ls='', color=color)
        ax.semilogy(delta_time_3, exposure_sec_3, ls=ls, color=color)
        ax.set_title(self._dataset_name)
        ax.set_ylabel("exposure")
        ax.yaxis.label.set_color(color)

        ax.set_xlim(xlims)
        f.tight_layout()
        if legend:
            ax.legend()

    def show(self):
        for f in self._figs:
            f.show()
