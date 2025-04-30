import logging

import matplotlib
from matplotlib import pyplot as plt

import procastro as pa
from procastro import calib
from procastro.exceptions import EmptyAstroDirError
from procastro.logging import io_logger

matplotlib.use('TkAgg')
io_logger.setLevel(logging.DEBUG)

raw_wavpix_file = r"C:\Users\duckr\Documents\astro-data\imacs\aug24\recal\ob{trace:02d}_lines_chips.txt"
wavpix_file = r"C:\Users\duckr\Documents\astro-data\imacs\aug24\recal\wavpix_{trace:02d}_v{version:d}.ecsv"
spec_file = r"C:\Users\duckr\Documents\astro-data\imacs\aug24\recal\ob{trace:02d}_{chip:d}_spec.fits"
arcs_file = r"C:\Users\duckr\Documents\astro-data\imacs\aug24\recal\ob{trace:02d}_{chip:d}_arc_{element:s}.fits"

wavpix_file = r"out\wavpix_{trace:02d}_v{version:d}.ecsv"

use_mosaic = calib.WavMosaic()

##################
# Preparing the "wavelength identification" calibration from arc lamps

try:
    wavpix = pa.AstroDir(wavpix_file, filter_and={'version': 0}, group_by='trace')
    wav_solution = calib.WavSol(wavpix, beta=8, wav_out=2048, align_telluric='7')
except EmptyAstroDirError:
    io_logger.warning("Recomputing wavelength-pixel fitting")
    label_col = calib.TableOp(add={'label': 'np.char.add([element].astype(str), np.int32([wav]).astype(str))'})
    dtype={"names": ('wav', 'pix', 'chip', 'element'), 'formats': ('f', 'f', 'i', 'S2')}

    wavpix = pa.AstroDir(raw_wavpix_file,
                         astrocalib=[label_col, use_mosaic],
                         file_format=f"loadtxt.{dtype}",
                         meta={'version': 0, 'instrmnt': 'imacs'},
                         )
    arcs = pa.AstroDir(arcs_file,
                       astrocalib = use_mosaic,
                       meta={'version': 0, 'instrmnt': 'imacs'}
                       ).mosaic_by('trace', 'element')

    wav_solution = calib.WavSol(wavpix, beta=8, wav_out=2048, align_telluric='7')
    wav_solution.plot_residuals(title="pre-refit")
    wav_solution.add_arc(arcs, refit=False)
    wav_solution.plot_fit(title="pre")
    wav_solution.refit(beta=8)
    wav_solution.plot_fit(title="post")
    wav_solution.plot_width()
    wav_solution.save_in("out", pattern="wavpix_{trace:02d}_v{version:d}.ecsv")

wav_solution.plot_residuals(title="post-refit")


#################
# reading science files into SpecMosaic  (group_by keyword forces the mosaic rather than auto-identify)

spec = pa.AstroDir(spec_file,
                   astrocalib=use_mosaic,
                   meta={'instrmnt': 'imacs'},
                   ).mosaic_by('trace')

###################
# plotting before and after applying wav_solution for frame #6 (randomly chosen)

spec[6].plot(channels=7, epochs=range(0,100,10), title="pre")

spec.add_calib(wav_solution)
spec[6].plot(channels=7, epochs=range(100), title="post")

###################
# Save calibrated frames

spec.save_in("out", filename_pattern="spec{trace:02d}.fits")

plt.show()
