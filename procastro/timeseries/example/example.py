import procastro as pa
import procastro.core.astrofile.astrofile
import procastro.timeseries as tm

raw = pa.AstroDir("./data/raw")

dark = procastro.core.astrofile.astrofile.AstroFile("./data/dark.fits")
flat = procastro.core.astrofile.astrofile.AstroFile("./data/flat.fits")

target_coords = [[570, 269], [436, 539]] # Coordinates of 1 target and 1 reference
# Coordinates in y, x format!!!
labels = ["Target", "Ref1"]
aperture = 12  # Aperture
sky = [16, 20]  # Sky
stamp_rad = 30  # Square radius for stamp

# These values are made up!
ron = 1.
gain = 1.

# ======= INITIALIZING PHOTOMETRY OBJECT =======
# Initialize Photometry object. If calculate_stamps=True, stamps for photometry will be calculated upon
# initialization and get_stamps does not have to be explicitely called by the user
phot = tm.Photometry(raw, aperture=aperture, sky=sky, mdark=dark, mflat=flat,
                     arget_coords_xy=target_coords,
                     stamp_rad=stamp_rad,
                     labels=labels, gain=gain, ron=ron)

# Plot radial profile of targets; can specify targets with labels
phot.plot_radialprofile()
phot.plot_radialprofile(targets=['Ref1'])
# Can also specify for which frame to plot
phot.plot_radialprofile(targets=['Ref1', 'Target'], frame=2)

# Plot stamps used for photometry, first and last frames to be used can be given
phot.showstamp()
phot.showstamp(first=0, last=5)

# ======= EXECUTING PHOTOMETRY =======
# Aperture and sky are optional, they can be given when initializing Photometry object or when running
# the photometry function. The GPU flag is also optional (default is gpu=False). The results are
# TimeSerie.TimeSeries objects!
ts_cpu = phot.photometry(aperture=aperture, sky=sky)
ts_gpu = phot.photometry(gpu=True)

print(ts_cpu.__class__)  # Just to show that it is a TimeSerie.TimeSeries object

# Can be run without reducing, ie without bias, dark, flat
phot2 = Photometry(raw, aperture, sky, mdark=None, mflat=None,
                  calculate_stamps=True, target_coords_xy=target_coords, stamp_rad=stamp_rad,
                  labels=labels, gain=gain, ron=ron)

ts_cpu2 = phot2.photometry()


# The following should be done in case the user wants to explicitely get the data stamps
# It should NOT be necessary to run the program this way as get_stamps is called inside photometry
# But added to example just in case
sci_stamps, centroid_coords, stamp_coords, epoch, labels = phot.get_stamps(raw, target_coords, stamp_rad)
phot3 = Photometry(sci_stamps, aperture, sky, mdark=dark, mflat=flat, calculate_stamps=False,
                   target_coords_xy=target_coords, stamp_rad=stamp_rad, new_coords=centroid_coords,
                   stamp_coords=stamp_coords, epoch=epoch, labels=labels, gain=gain, ron=ron)

ts_cpu_3 = phot3.photometry()

# ======= OPERATIONS FOR TIMESERIES =======
# Plot photometry results; ts_cpu and ts_gpu are TimeSerie.TimeSeries objects
ts_cpu.plot()
ts_gpu.plot()

# Access to light curves for different targets
res_ref1_cpu = ts_cpu[1]  # Returns the light curve of the 2 nd target object as scipy array
res_ref1_gpu = ts_gpu['Ref1']  # Returns the light curve of the target assigned the 'REF1' label

# Get the error corresponding to the data channel
ref1_error_cpu = ts_cpu.get_error(1)
target_error_cpu = ts_cpu.get_error('Target')

# TimeSerie channels can be grouped to perform operations between them.
# Only two groups are allowed for each TimeSerie object. The default grouping is
# the first target in a group, and the rest of the targets in another group. To group channels
# differently, a mask must be applied (error channels are also grouped according to the mask):
ts_cpu.set_group([1, 1, 0, 1, 1])  # Groups channels 0, 1, 3, and 4 in one group, 2 in another
ts_gpu.set_group([1, 1, 1, 0, 0])  # Groups channels 0, 1, and 2 in one group, 3 and 4 in another

# Retrieve grouped channels:
group1_cpu = ts_cpu.group1()
group1_cpu_errors = ts_cpu.errors_group1()

# Calculate mean or median of a group
ts_cpu.mean(1)
ts_gpu.median(2)

# Results of these operations are stored in the "hidden" channels ts[-group_id]
res_cpu_mean = ts_cpu[-1]
res_gpu_median = ts_gpu[-2]

# Operations can be performed between individual channels. The results of these operations
# are NOT stored in the TimeSeries object, only returned
div = ts_cpu[1]/ts_cpu[0]
added = ts_gpu[0] + ts_gpu['Ref1']






