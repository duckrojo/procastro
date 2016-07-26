#
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

import types

modules = ['photometry']
           
for modulename in modules:
    module = __import__(modulename, globals(), locals(), [], -1)
    module = reload(module)
    for v in dir(module):
        if v[0] == '_' or isinstance(getattr(module,v), types.ModuleType):
            continue
        globals()[v] = getattr(module, v)
    del module

del modules, modulename, types




if __name__ == '__main__':

    # Initialization using directory path
    print '\n*************************************************************'
    print 'Initialization using directory path and providing coordinates'
    print '*************************************************************\n'

    # Data from: http://wasabi.das.uchile.cl/wasp/
    dir_path = '/home/fcaro/Dropbox/Pato/data'
    cooxy = [[226, 633], [288, 292], [861, 735]]
    labels = ['Star_0', 'Star_1', 'Star_2']

    print 'Beginning Timeseries instance initialization\n'
    TS = Timeseries(dir_path, cooxy, labels=labels)
    print 'Timeseries instance initializated'

    print '\nPerforming aperture photometry'
    TS_out_1 = TS.perform_phot(5, [10, 13])
    TS_out_2 = TS.perform_phot(5)
    print 'Aperture photometry finished'

    print '\nBeginning plot displaying'
    TS_out_1.plot_timeseries()
    TS_out_2.plot_timeseries()
    TS_out_1.plot_ratio(trg=labels[2])
    TS_out_2.plot_ratio(trg=labels[2])
    TS_out_1.plot_drift()
    TS_out_2.plot_drift()
    print 'Plot displaying finished'

    # Initialization using astrodir
    print '\n******************************************************'
    print 'Initialization using astrodir and interactive mode (no'
    print 'coordinates provided)'
    print '******************************************************\n'

    import dataproc as dp
    adir = dp.astrodir(dir_path)

    print 'Beginning Timeseries instance initialization\n'
    TS = Timeseries(adir)  # NO COORDINATES PROVIDED
    print 'Timeseries instance initializated'

    print '\nPerforming aperture photometry'
    TS_out_1 = TS.perform_phot(5, [10, 13])
    TS_out_2 = TS.perform_phot(5)
    print 'Aperture photometry finished'

    print '\nBeginning plot displaying'
    TS_out_1.plot_timeseries()
    TS_out_2.plot_timeseries()
    TS_out_1.plot_ratio()
    TS_out_2.plot_ratio()
    TS_out_1.plot_drift()
    TS_out_2.plot_drift()
    print 'Plot displaying finished'

    # Initialization using a list of 10 images (array list)
    print '\n********************************************************'
    print 'Initialization using a list of 10 images and interactive'
    print 'mode (no coordinates provided)'
    print '********************************************************\n'
    import pyfits as pf
    import os

    file_list = [
        'obj9020.fits', 'obj9021.fits', 'obj9022.fits', 'obj9023.fits', 'obj9024.fits',
        'obj9025.fits', 'obj9026.fits', 'obj9027.fits', 'obj9028.fits', 'obj9029.fits']

    img_list = []
    for filename in file_list:
        arch = pf.open(os.path.join(dir_path, filename))
        img_list.append(arch[0].data)

    print 'Beginning Timeseries instance initialization\n'
    TS = Timeseries(img_list)  # NO COORDINATES PROVIDED
    print 'Timeseries instance initializated'

    print '\nPerforming aperture photometry'
    TS_out_1 = TS.perform_phot(5, [10, 13])
    TS_out_2 = TS.perform_phot(5)
    print 'Aperture photometry finished'

    print '\nBeginning plot displaying'
    TS_out_1.plot_timeseries()
    TS_out_2.plot_timeseries()
    TS_out_1.plot_ratio()
    TS_out_2.plot_ratio()
    TS_out_1.plot_drift()
    TS_out_2.plot_drift()
    print 'Plot displaying finished'
