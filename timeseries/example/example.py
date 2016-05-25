from dataproc.core import AstroFile, AstroDir
from dataproc.timeseries.stamp_photometry import Photometry

raw = AstroDir("./data/raw")

dark = AstroFile("./data/dark.fits")
flat = AstroFile("./data/flat.fits")

target_coords = [[570, 269], [436, 539]] # Coordinates of 1 target and 1 reference
labels = ["Target", "Ref1"]
aperture = 12  # Aperture
sky = [16, 20]  # Sky
stamp_rad = 30  # Square radius for stamp

# These values are made up!
ron = 1.
gain = 1.

# Initialize Photometry object. It calculate_stamps=True, stamps for photometry will be calculated upon
# initialization and get_stamps does not have to be explicitely called by the user
phot = Photometry(raw, aperture, sky, mdark=dark, mflat=flat,
                  calculate_stamps=True, target_coords=target_coords, stamp_rad=stamp_rad,
                  labels=labels, gain=gain, ron=ron)

ts_cpu = phot.photometry()
ts_gpu = phot.photometry(gpu=True)

print(ts_cpu.__class__)  # Just to show that it is a TimeSerie.TimeSeries object

ts_cpu.plot()
ts_gpu.plot()

# Can be run without reducing, ie without bias, dark, flat
phot2 = Photometry(raw, aperture, sky, mdark=None, mflat=None,
                  calculate_stamps=True, target_coords=target_coords, stamp_rad=stamp_rad,
                  labels=labels, gain=gain, ron=ron)

ts_cpu2 = phot2.photometry()


# The following should be done in case the user wants to explicitely get the data stamps
# It should NOT be necessary to run the program this way as get_stamps is called inside photometry
# But added to example just in case
sci_stamps, centroid_coords, stamp_coords, epoch, labels = phot.get_stamps(raw, target_coords, stamp_rad)
phot3 = Photometry(sci_stamps, aperture, sky, mdark=dark, mflat=flat, calculate_stamps=False,
                   target_coords=target_coords, stamp_rad=stamp_rad, new_coords=centroid_coords,
                   stamp_coords=stamp_coords, epoch=epoch, labels=labels, gain=gain, ron=ron)

ts_cpu_3 = phot3.photometry()
