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

import glob
import astropy.units as u
import astropy.time as apt
import exifread as er
import numpy as np
import pandas as pd
import dataproc as dp

__all__ = ['RawFiles', 'RawFile']


class RawFiles:
    def __init__(self, dirname=None, name="", ref_time=None):
        self._df = pd.DataFrame()

        if isinstance(ref_time, str):
            ref_time = apt.Time(ref_time)
        self._ref_time = ref_time

        image_list = glob.glob(dirname+"/*nef")
        self._dataset_name = name
        self._raw_files = [RawFile(image) for image in image_list]

    def __len__(self):
        return len(self._raw_files)

    def __repr__(self):
        return f"<dataproc.photography: {len(self)} RawFiles>"

    def __getitem__(self, item):
        return self._raw_files[item]

    def read_exif(self, **kwargs):
        i = 0

        print(f"Processing {len(self._raw_files)} images: ", end='')
        for image in self._raw_files:
            data_read = image.read_exif(**kwargs)
            self._df = self._df.append(data_read, ignore_index=True)
            if i % 10 == 0:
                if i % 100 == 0:
                    print(f'\n{i:3d}', end='')
                else:
                    print(f'{(i//10) % 10}', end='')
            else:
                print(".", end="")
            i += 1
        print("")

        self._df['ev'] = np.log(100 * self._df['fnumber'] ** 2
                                / self._df['exposure'] / self._df['iso']) / np.log(2)

        return self

    def set_ref_time(self, ref_time=None, ref_frame=0):
        df = self._df
        if ref_time is None:
            ref_time = self._ref_time
        if ref_time is None:
            ref_time = df['date'][ref_frame]

        df['delta_time'] = [(dt - ref_time).to(u.s).value for dt in df['date']]
        df['delta_post'] = df['delta_time'] + df['exposure']

    def plot_ev(self, ref_frame=0, ref_time=None,
                marker='.', ls='', color='red',
                label=None, legend=False,
                xlims=None, ax=None):
        self.set_ref_time(ref_time, ref_frame)
        df = self._df
        f, ax = dp.prep_canvas(ax)

        ax.plot(df['delta_time'], df['ev'], label=label,
                marker=marker, ls=ls, color=color)
        ax.set_title(self._dataset_name)
        ax.set_ylabel("ev")
        ax.yaxis.label.set_color(color)

        ax.set_xlim(xlims)
        f.tight_layout()
        if legend:
            ax.legend()
        f.show()

    def plot_exposure(self, ref_time=None, ref_frame=0,
                      marker='|', ls='-', color='blue',
                      label=None, legend=False,
                      xlims=None, ax=None):
        self.set_ref_time(ref_time, ref_frame)
        df = self._df
        f, ax = dp.prep_canvas(ax)

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
        f.show()


class RawFile:
    def __repr__(self):
        return f"RAW image {self._filename}"

    def __init__(self, filename):
        self._filename = filename
        self.data = None

    def read_exif(self, fields=None, reload=False, fnumber_def=None):
        if self.data is not None and not reload:
            return self.data

        if fields is None:
            fields = {'EXIF FNumber': 'fnumber_str',
                      'EXIF ExposureTime': 'exposure_str',
                      'EXIF ISOSpeedRatings': 'iso_str',
                      'EXIF DateTimeOriginal': 'date_str'}

        tags = er.process_file(open(self._filename, 'rb'))
        data = {v: tags[f].printable for f, v in fields.items()}

        data['exposure'] = eval(data['exposure_str'])
        data['iso'] = eval(data['iso_str'])
        if fnumber_def is None:
            fnumber_def = data['fnumber_str']
        data['fnumber'] = eval(fnumber_def)
        data['date'] = apt.Time(data['date_str'].replace(":", "-", 2))

        self.data = data
        return data


# df = pd.DataFrame()
#
# dirname = "d:/dcim/102ND750/"
# image_list = glob.glob(dirname+"/*nef")
#
# i = 0
# print(f"Procesing {len(image_list)} images: ", end='')
# for image in image_list:
#     tags = er.process_file(open(image, 'rb'))
#     fields = ['EXIF FNumber', 'EXIF ExposureTime',
#               'EXIF ISOSpeedRatings', 'EXIF DateTimeOriginal']
#     vals = [tags[f].printable for f in fields]
#
#     data = {'fnumber': eval(vals[0]),
#             'exposure': vals[1],
#             'exposure_sec': eval(vals[1]),
#             'iso': eval(vals[2]),
#             'date': apt.Time(vals[3].replace(":", "-", 2))
#             }
#     df = df.append(data, ignore_index=True)
#     if i % 10 == 9:
#         print('x', end='')
#     else:
#         print(".", end="")
#     i += 1
# print("")
#
# reftime = df['date'][0]
#
# df['delta_time'] = [(dt-reftime).to(u.s).value for dt in df['date']]
# df['delta_post'] = df['delta_time'] + df['exposure_sec']
# df['ev'] = np.log(100*df['fnumber']**2/df['exposure_sec']/df['iso'])/np.log(2)
#
# delta_time_2 = [item for sublist in zip(df['delta_time'], df['delta_post']) for item in sublist]
# exposure_sec_2 = [item for item in df['exposure_sec'] for _ in (1, 2)]
#
# f, ax = plt.subplots(1, 1)
# ax.plot(df['delta_time'], df['ev'], '.r')
# ax2 = ax.twinx()
# ax2.semilogy(delta_time_2, exposure_sec_2, color='b')
# ax.set_ylabel("ev")
# ax.yaxis.label.set_color('red')
# ax2.set_ylabel("exposure")
# ax2.yaxis.label.set_color('blue')
#
# f.tight_layout()
# plt.show()
