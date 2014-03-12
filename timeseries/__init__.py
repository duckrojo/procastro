from timeseries import Timeseries

if __name__ == '__main__':

    # Initialization using directory path
    print 'Initialization using directory path'

    dir_path = '/home/fcaro/Dropbox/Timeseries/data'
    coo = [[226, 633], [288, 292], [861, 735]]
    labels = ['Star1', 'Star2', 'Star3']

    TS = Timeseries(dir_path, coo, labels=labels)

    TS_out_1 = TS.perform_phot(5, [10, 13])
    TS_out_2 = TS.perform_phot(5)

    TS_out_1.plot_timeseries()
    TS_out_2.plot_timeseries()

    TS_out_1.plot_ratio()
    TS_out_2.plot_ratio(trg=labels[1])

    TS_out_1.plot_drift()
    TS_out_2.plot_drift()

    # Initialization using astrodir
    print 'Initialization using astrodir'
    import dataproc as dp

    adir = dp.astrodir(dir_path)
    TS = Timeseries(adir, coo, labels=labels)

    TS_out_1 = TS.perform_phot(5, [10, 13])
    TS_out_2 = TS.perform_phot(5)

    TS_out_1.plot_timeseries()
    TS_out_2.plot_timeseries()

    TS_out_1.plot_ratio()
    TS_out_2.plot_ratio(trg=labels[1])

    TS_out_1.plot_drift()
    TS_out_2.plot_drift()

    # Initialization using a list of 10 images (array list)
    print 'Initialization using a list of 10 images'
    import pyfits as pf
    import os

    file_list = [
        'obj9020.fits', 'obj9021.fits', 'obj9022.fits', 'obj9023.fits', 'obj9024.fits',
        'obj9025.fits', 'obj9026.fits', 'obj9027.fits', 'obj9028.fits', 'obj9029.fits']

    img_list = []
    for filename in file_list:
        arch = pf.open(os.path.join(dir_path, filename))
        img_list.append(arch[0].data)

    TS = Timeseries(img_list, coo, labels=labels)

    TS_out_1 = TS.perform_phot(5, [10, 13])
    TS_out_2 = TS.perform_phot(5)

    TS_out_1.plot_timeseries()
    TS_out_2.plot_timeseries()

    TS_out_1.plot_ratio()
    TS_out_2.plot_ratio(trg=labels[1])

    TS_out_1.plot_drift()
    TS_out_2.plot_drift()
