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
import astropy.time as apt
import astropy.units as u
import exifread as er
import numpy as np
import pandas as pd
from .plotting import PhotoPlotting

__all__ = ['RawFiles', 'RawFile']


class RawFiles(PhotoPlotting):
    def __init__(self,
                 dirname=None, name="", ref_time=None,
                 hour_offset=0,
                 extension="nef"):
        super().__init__()
        self._df = pd.DataFrame()

        if isinstance(ref_time, str):
            ref_time = apt.Time(ref_time)
        ref_time += hour_offset*u.h
        self._ref_time = ref_time

        image_list = glob.glob(f"{dirname}/*{extension}")
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
