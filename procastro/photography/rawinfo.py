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
import rawpy as rp


__all__ = ['RawFiles', 'RawFile']


class RawFiles(PhotoPlotting):
    def __init__(self,
                 dirname,
                 umbra=40,
                 name="", ref_time=None,
                 min_offset=0, fnumber_def=None,
                 extension="nef", sort_time=True,
                 verbose=True):
        """
Container of image information class
        Args:
            dirname: str, (list, tuple)
              Directory where to look for files, or list of files
            name: str
               Name of the dataset
            ref_time: astropy.time.Time, int, str
               Frame to take as reference for time, or string that can be converted to time, or time object
            min_offset: float
               Offset of time in hours
            extension: str
                Extension of files to consider
        """
        super().__init__(umbra=umbra)
        database = pd.DataFrame()
        self._df = database
        self._extension = extension

        if isinstance(dirname, (list, tuple)):
            self._raw_files = dirname
            error_msg = f"Empty list given to RawFiles"
        else:
            search_string = f"{dirname}/*{extension}"
            image_list = glob.glob(search_string)
            self._raw_files = [RawFile(image, fnumber_def=fnumber_def)
                               for image in image_list]
            error_msg = f"No files found on: '{search_string}'"
        if not len(self._raw_files):
            raise ValueError(error_msg)

        self._dataset_name = name

        self.read_exif()

        if ref_time is None:
            ref_time = self._df['date'][0]
        elif isinstance(ref_time, str):
            ref_time = apt.Time(ref_time)
        ref_time += min_offset * u.min
        self._ref_time = ref_time

    def __len__(self):
        return len(self._raw_files)

    def __repr__(self):
        return f"<procastro.photography: {len(self)} RawFiles>"

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._raw_files[item]

        files = self._raw_files[item]
        return RawFiles(files, name=self._dataset_name,
                        ref_time=self._ref_time, extension=self._extension,
                        verbose=False)

    def read_exif(self, sort_time=True, verbose=True, **kwargs):
        i = 0

        if verbose:
            print(f"Processing {len(self._raw_files)} images: ", end='')
        for image in self._raw_files:
            data_read = image.read_exif(**kwargs)  # receives a dictiionary
            self._df = self._df.append(data_read, ignore_index=True)  # adds that info into database

            if verbose:
                if i % 10 == 0:
                    if i % 100 == 0:
                        print(f'\n{i:3d}', end='')
                    else:
                        print(f'{(i//10) % 10}', end='')
                else:
                    print(".", end="")
                i += 1
        if verbose:
            print("")

        if sort_time:
            self._df.sort_values('date', inplace=True)

        self._df['ev'] = np.log(100 * self._df['fnumber'] ** 2
                                / self._df['exposure'] / self._df['iso']) / np.log(2)

        return self

    def __iter__(self):
        for image in self._raw_files:
            yield image


class RawFile:
    def __repr__(self):
        return f"RAW image {self._filename}"

    def __init__(self, filename, fnumber_def=None):
        self._filename = filename
        self.header = None
        self.read_exif(fnumber_def=fnumber_def)

    def read_exif(self, fields=None, reload=False, fnumber_def=None):
        if self.header is not None and not reload:
            return self.header

        if fields is None:
            fields = {'EXIF FNumber': 'fnumber_str',
                      'EXIF ExposureTime': 'exposure_str',
                      'EXIF ISOSpeedRatings': 'iso_str',
                      'EXIF DateTimeOriginal': 'date_str'}

        tags = er.process_file(open(self._filename, 'rb'))
        header = {v: tags[f].printable for f, v in fields.items()}

        header['exposure'] = eval(header['exposure_str'])
        header['iso'] = eval(header['iso_str'])
        if fnumber_def is None:
            fnumber_def = header['fnumber_str']
        header['fnumber'] = eval(fnumber_def)
        header['date'] = apt.Time(header['date_str'].replace(":", "-", 2))
        header['ev'] = np.log(100 * header['fnumber'] ** 2
                              / header['exposure'] / header['iso']) / np.log(2)

        self.header = header
        return header

    @property
    def data(self):
        with rp.imread(self._filename) as raw:
            rgb = raw.postprocess()
            return rgb


#sample = "C:/Users/duckr/Desktop/post/sony/DSC09845.ARW"