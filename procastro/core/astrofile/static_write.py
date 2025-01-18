from astropy.io import fits as pf
from astropy import time as apt
import procastro as pa


def static_write(file_type, filename, data, meta,
                 overwrite=False,
                 ):
    match file_type:
        case "FITS":
            header = pf.Header(meta)
            header['history'] = "Saved by procastro v{} on {}".format(pa.__version__,
                                                                      apt.Time.now())

            return pf.writeto(filename, data, header,
                              overwrite=overwrite)

        case "ARRAY":
            raise TypeError("File type 'ARRAY' cannot save to file. Specify file_type explicitly")

    raise TypeError(f"File type {file_type} cannot be written.")
