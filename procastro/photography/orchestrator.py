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

__all__ = ['ReadScript']

import csv
import numpy as np
import pandas as pd
from .plotting import PhotoPlotting
import astropy.time as apt
import astropy.units as u


def hms_to_sec(hms):
    hh, mm, ss = (float(i) for i in hms.split(':'))
    return ss+60*(mm+60*hh)


class ReadScript(PhotoPlotting):
    def __init__(self, csv_file,
                 umbra=38.5, penumbra=31.1 + 60 * (43 + 60 * 1),
                 ref_time=None, name=None):
        super().__init__()
        self.C = ["", -(penumbra-umbra)/2, 0, umbra, (umbra+penumbra)/2]
        self._df = pd.DataFrame()
        self.penumbra = penumbra
        self.umbra = umbra
        self._dataset_name = name

        if ref_time is None:
            ref_time = apt.Time.now()

        print(f"Parsing orchestrator... ", end='')
        self._read_csv(csv_file)
        self.set_zero_time(ref_time)
        self.set_ref_time(ref_time=ref_time)
        print("done.")

    def set_zero_time(self, time_zero):
        if isinstance(time_zero, str):
            time_zero = apt.Time(time_zero)
        self._df['date'] = self._df['start']*u.s + time_zero
        self._df = self._df.sort_values('date')

    def _read_csv(self, csv_file):
        with open(csv_file) as f:
            for row in csv.reader(f):
                if row[0] == 'FOR':
                    dummy1, action, starting, step, n_iter = row
                    instructions = []
                    for row_for in csv.reader(f):
                        if row_for[0] == 'ENDFOR':
                            break
                        instructions.append(row_for)
                    assert len(instructions) == 1
                    if action == '(INTERVALOMETER)':
                        starting = (starting == '1')
                        n_iter = eval(n_iter)
                        step = eval(step)
                    if action == '(VAR)':
                        start, step, finish = eval(starting), eval(step), eval(n_iter)
                        n_iter = int((finish-start)/step)
                        step = (self.penumbra - self.umbra)/n_iter/2
                        starting = instructions[0][1][3:6] == 'POS'

                    ref = 0
                    for _ in range(n_iter):
                        ref = self.loop_exec_instruction(instructions[0],
                                                         ref=ref, step=step,
                                                         starting=starting)
                else:
                    ref = self.exec_instruction(row, ref=ref)

        self._df['ev'] = np.log(100 * self._df['fnumber'] ** 2
                                / self._df['exposure'] / self._df['iso']) / np.log(2)

    def exec_instruction(self, row, ref=0):
        if row[0] == 'TAKEPIC':
            (action, ref_str, sign, offset_single, camera, shutter,
             fnumber, iso, dummy, filetype, MLU, increment) = row[0:12]
            comment = ",".join(row[12:])

            if ref_str[0] == 'C':
                ref = self.C[int(ref_str[1])]
            elif ref_str == 'MAX':
                ref = (self.C[2] + self.C[3]) / 2
            elif ref_str == 'LAST':
                pass
            else:
                raise ValueError(f"reference {ref_str} not understood")
            offset = hms_to_sec(offset_single)
            ref += eval(f"{sign}1")*offset
            duration = eval(shutter)

            self.add_to_dataframe(ref, comment, shutter, fnumber, iso)
            return ref

        elif row[0] == 'PLAY':
            return ref
        else:
            raise ValueError(f"Instruction '{row[0]}' not understood!")

    def loop_exec_instruction(self, row, ref=0, step=0, starting=True):
        if row[0] == 'TAKEPIC':
            (action, ref_str, sign, offset_single, camera, shutter,
             fnumber, iso, dummy, filetype, MLU, increment) = row[0:12]
            comment = ",".join(row[12:])

            if ref == 0:
                if ref_str[3:6] == 'PRE':
                    ref = self.C[2] - step
                elif ref_str[3:6] == 'POS':
                    ref = self.C[3] + step
                elif ref_str[0] == 'C':
                    ref += self.C[int(ref_str[1])]
                elif ref_str == 'MAX':
                    ref += (self.C[2] + self.C[3]) / 2
                ref += eval(f"{sign}1")*hms_to_sec(offset_single)
            duration = eval(shutter)
            if starting:
                start = ref
                return_offset = start + duration + step
            else:
                start = ref - duration
                return_offset = ref - duration - step

            self.add_to_dataframe(start, comment, shutter, fnumber, iso)
            return return_offset

        elif row[0] == 'PLAY':
            return ref
        else:
            raise ValueError(f"Instruction '{row[0]}' not understood!")

    def add_to_dataframe(self, start, comment, shutter, fnumber, iso):
        data = {'start': start,
                'comment': comment,
                'exposure_str': shutter,
                'fnumber_str': fnumber,
                'iso': eval(iso),
                }
        data['exposure'] = eval(data['exposure_str'])
        data['fnumber'] = eval(data['fnumber_str'])
        self._df = self._df.append(data, ignore_index=True)


if __name__ == '__main__':
    import procastro.photography as pap
    import astropy.units as u

    basedir = "C:/Users/duckr/OneDrive/Documents/eclipse/rxs/"
    files = ["1_0+4_0", "1_0+2_0",
             "0_7+1_6+x3", "0_8+1_6+x3+2",
             "0_9+1_6+x3+2_3", "1_0+1_6+x3+2_5",
             "r1_0+1_6+x3+2_5",
             ]
    ref_time = ["2021-11-20 01:33", "2021-11-20 17:01",
                "2021-11-20 17:54", "2021-11-20 18:09:37",
                "2021-11-20 18:16:04", "2021-11-20 18:36:02",
                "2021-11-20 18:44",
                ]

    item = 6

    out = pap.ReadScript(f"{basedir}{files[item]}/rxs.csv",
                         ref_time=ref_time[item],
                         name=files[item])

    out.plot_ev(xlims=[-15, 50], ax=1, marker='v',
                label="Orchestrator")
    # out.plot_exposure(xlims=[-20, 60], ax=2)

    raw = pap.RawFiles(f"{basedir}{files[item]}",
                       name=files[item],
                       hour_offset=-0.052, extension="cr2",
                       ref_time=ref_time[item]).read_exif()
    raw.plot_ev(ax=1, marker='^', color='blue', legend=True,
                label="Camera")
    # raw.plot_exposure(ax=2)

    # raw.show()
