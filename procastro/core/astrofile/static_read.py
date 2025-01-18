from astropy.io import fits as pf


def static_read(file_type, filename):

    match file_type:
        case "FITS":
            elements = filename.split(":")

            if len(elements) == 1:
                hdu = 0
            else:
                hdu = int(elements[1])

            unit = pf.open(filename)[hdu]

            return unit.data, unit.header

        case "ARRAY":
            return filename

    raise TypeError(f"File type {file_type} cannot be read.")
