import matplotlib

import procastro as pa
from procastro import calib

matplotlib.use('TkAgg')

wavpix_file = r"C:\Users\duckr\Documents\prg\astro\exoatm\imacs\aug24\recal\wavpix_{trace:02d}_v{version:d}.ecsv"
spec_file = r"C:\Users\duckr\Documents\prg\astro\exoatm\imacs\aug24\recal\ob{trace:02d}_{chip:d}_spec.fits"


##################
# Preparing the "wavelength identification" calibration from arc lamps

wavpix_file = r"out\wavpix_{trace:02d}_v{version:d}.ecsv"
wavpix = pa.AstroDir(wavpix_file, filter_and={'version': 0}, group_by='trace')
wav_solution = calib.WavSol(wavpix, beta=8, wav_out=2048, align_telluric='7')
wav_solution.plot_residuals()


#################
# reading science files into SpecMosaic  (group_by keyword forces the mosaic rather than auto-identify)

imacs_f2_offset = {4: 0,
                   7: 35 + 2048,
                   3: 0,
                   8: 35 + 2048,
                   2: 0,
                   6: 35 + 2048,
                   1: 0,
                   5: 35 + 2048,
                   }
spec = pa.AstroDir(spec_file,
                   astrocalib=calib.WavMosaic(imacs_f2_offset),
                   group_by='trace')


###################
# plotting before and after applying wav_solution for frame #6 (randomly chosen)

spec[6].plot(channels=7, epochs=range(100), title="pre")
spec.add_calib(wav_solution)
spec[6].plot(channels=7, epochs=range(100), title="post")


###################
# Save calibrated frames

spec.save_in("out", filename_pattern="spec{trace:02d}_ch{channel}.fits")
spec[0].write_as("out/ wav.fits", channels='wav')
