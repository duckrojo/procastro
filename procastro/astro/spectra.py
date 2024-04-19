import glob
import warnings

import numpy as np
from astropy import units as u, constants as c

import procastro as pa
import os.path as path


def blackbody(temperature, wav_freq, unit=None):
    """
    Computes a blackbody curve for the parameters given

    Parameters
    ----------
    temperature : astropy.unit
        Assumes kelvin if unit is not specified
    wav_freq : astropy.unit
        Wavelength or frequency
    unit : astropy.unit, optional
        unit for wav_freq from astropy.unit. If None and wav_freq is not an
        astropy.quantity, then assume microns

    Returns
    -------
    Blackbody curve at specified wavelength or frequency

    Raises
    ------
    ValueError
        wav_frequency unit is invalid
    """

    if not isinstance(temperature, u.Quantity):
        temperature = temperature * u.K
    if not isinstance(wav_freq, u.Quantity):
        if unit is None:
            wav_freq = wav_freq * u.micron
        elif isinstance(unit, u.Unit):
            wav_freq = wav_freq * unit
        else:
            raise ValueError("Specified unit ({0:s}) is not a valid "
                             "astropy.unit".format(unit,))

    if wav_freq.cgs.unit == u.cm:
        use_length = True
    elif wav_freq.cgs.unit == 1 / u.s:
        use_length = False
    else:
        raise ValueError("Units for x must be either length or frequency, "
                         "not {0:s}".format(wav_freq.unit,))

    h_kb_t = c.h / c.k_B / temperature

    if use_length:
        blackbody_return = 2 * c.h * c.c ** 2 / (wav_freq ** 5) / (np.exp(h_kb_t * c.c / wav_freq) - 1)
        blackbody_return = blackbody_return.to(u.erg / u.cm ** 2 / u.cm / u.s) / u.sr
    else:
        blackbody_return = 2 * c.h * wav_freq ** 3 / (c.c ** 2) / (np.exp(h_kb_t * wav_freq) - 1)
        blackbody_return = blackbody_return.to(u.erg / u.cm ** 2 / u.Hz / u.s) / u.sr

    return blackbody_return


def getfilter(name,
              name_only=False,
              filter_unit=None,
              fct=1,
              filter_dir='/home/inv/common/standards/filters/oth/*/',
              force_positive=True,
              force_increasing=True,
              ):
    """
    Get transmission curve for a particular filter

    Parameters
    ----------
    name: str or array_like
        Name of the filter (filenames in directory). If array_like, then it
        needs to have two elements to define a uniform filter within the
        specified cut-in and cut-out values with value 1.0 and in a range 10%
        beyond with value 0.0, for a total of 50 pixels.
    name_only : bool, optional
        If True returns the name of the file only
    filter_unit : astropy quantity, optional
        Default filter unit that should be in files. Default is Angstroms.
        Can be specified in file as first line with comment 'unit:'
    fct : int, optional
        Factor by which the transmission is divided in the archive
        (i.e. 100 if it is written in percentage).
        Can be specified in file as first line with comment 'fct:'.
        Default is 1
    filter_dir : str, optional
        Directory that hold the filters, it accepts wildcards that will be used
        by glob. Default is ``/home/inv/common/standards/filters/oth/*/``
    force_positive : bool, optional
        Force positive values for transmission
    force_increasing : bool, optional
        Sort such that wavelength is always increasing

    Returns
    -------
    wavelength, transmission
        Transmission curve
    """

    filters = glob.glob(filter_dir + '/*.dat')

    if filter_unit is None:
        filter_unit = u.AA

    # if cut-in and cut-out values were specified
    if isinstance(name, (list, tuple)) and len(name) == 2:
        n_wavs = 50
        delta_filter = name[1] - name[0]
        axis = np.linspace(name[0] - 0.1*delta_filter,
                           name[1] + 0.1*delta_filter, n_wavs)
        transmission = np.zeros(n_wavs)
        transmission[(axis < name[1]) * (axis > name[0])] = 1
        return axis * filter_unit, transmission
    # otherwise, only accept strings
    elif not isinstance(name, str):
        raise TypeError(
            "Filter can be a string identifying its name or a two-element "
            "tuple identifying cut-in & cut-out values for an uniform filter")

    found = []
    for f in filters:
        if name in path.basename(f):
            found.append(f)

    new_line = '\n '
    if not len(found):
        raise ValueError(f"No filter matches '{name}'"
                         f" Be more specific between:\n {new_line.join(found):s}")
    if len(found) > 2:
        raise ValueError(f"Only one of the available filters should match '{name:s}'."
                         f" Be more specific between:\n {new_line.join(found):s}")

    if name_only:
        return path.basename(found[0])

    line1 = open(found[0]).readline()
    if line1[0].lower() == '#':
        items = line1[1:-1].split(',')
        for item in items:
            fld, val = item.split(':')
            if fld.lstrip() == 'units':
                filter_unit = getattr(u, val.lstrip())
            if fld.lstrip() == 'fct':
                fct = float(val.lstrip())

    axis, transmission = np.loadtxt(found[0], unpack=True)
    axis *= filter_unit
    transmission /= fct
    if force_positive:
        transmission[transmission < 0] = 0.0

    if force_increasing:
        axis, transmission = pa.sortmany(axis, transmission)
        axis = u.Quantity(axis)
        transmission = np.array(transmission)

    return axis, transmission


def applyfilter(name, spectra,
                wav_freq=None, n_wav_bb=100,
                output_unit=None,
                filter_dir=None,
                full=False):
    """
    Apply filter to spectra. It can be given or blackbody

    Parameters
    ----------
    name : str
      Filter name
    spectra: array_like or float
      Either flux (if iterable) or temperature (if scalar) for blackbody
    wav_freq: array_like, optional
      Wavelength or frequency. Only necessary when flux is given as spectra
    n_wav_bb: int, optional
      Number of wavelength points for blackbody. Default is 100
    output_unit : astropy.unit, optional
      Default unit for output. Default is nanometers
    filter_dir : str, optional
      If None use getfilter's default
    full : bool, optional
      If True, then return (filtered value, central weighted wavelength,
                                            equivalent width)

    Returns
    -------
    float
    """
    import scipy.interpolate as it

    filter_wav, filter_transmission = getfilter(name, filter_dir)
    filter_unit = filter_wav.unit
    if output_unit is None:
        output_unit = u.micron
    wav_min, wav_max = filter_wav[0], filter_wav[-1]
    us = it.UnivariateSpline(filter_wav, filter_transmission, s=0.0)

    try:
        if wav_freq is None:
            raise ValueError(
                "wav_freq needs to be specified if spectra is given")
        if not isinstance(wav_freq, u.Quantity):
            wav_freq *= filter_unit

        if wav_freq.cgs.unit == u.cm:
            if not isinstance(spectra, u.Quantity):
                spectra *= u.erg / u.cm ** 2 / u.cm / u.s / u.sr
        elif wav_freq.cgs.unit == 1 / u.s:
            if not isinstance(spectra, u.Quantity):
                spectra *= u.erg / u.cm ** 2 / u.Hz / u.s / u.sr
            spectra = spectra.to(u.erg / u.cm ** 2 / u.cm / u.s / u.sr,
                                 equivalencies=u.spectral_density(wav_freq))[::-1]
            wav_freq = wav_freq.to(u.nm,
                                   equivalencies=u.spectral())[::-1]
            print("WARNING: frequency domain filtering has not been tested "
                  "thoroughly!!")
        else:
            raise ValueError(
                "Invalid unit ({0:s}) for wav_freq. Currently supported: "
                "length and frequency".format(wav_freq.unit,))

        idx = (wav_freq > wav_min) * (wav_freq < wav_max)
        if len(idx) < n_wav_bb / 5:
            warnings.warn(f"Too little points ({len(idx)}) from given spectra "
                          f"inside the filter range ({wav_min} - {wav_max})")
        wav_freq = wav_freq[idx]
        spectra = spectra[idx]
    except TypeError:  # spectra is scalar (temperature)
        wav_freq = np.linspace(wav_min, wav_max, n_wav_bb)
        spectra = blackbody(spectra, wav_freq)

    spec_flt_tot = it.UnivariateSpline(wav_freq,
                                       (spectra * us(wav_freq)).value,
                                       s=0.0).integral(wav_freq[0].value,
                                                       wav_freq[-1].value)
    wav_flt_tot = it.UnivariateSpline(wav_freq,
                                      (wav_freq * us(wav_freq)).value,
                                      s=0.0).integral(wav_freq[0].value,
                                                      wav_freq[-1].value)
    flt_tot = us.integral(wav_freq[0].value,
                          wav_freq[-1].value)
    cwav = wav_flt_tot * filter_unit / flt_tot
    dwav = flt_tot * filter_unit / 2

    if full:
        return spec_flt_tot / flt_tot, cwav.to(output_unit), dwav.to(output_unit)
    else:
        return spec_flt_tot / flt_tot


def filter_conversion(target_filter, temperature,
                      temperature_ref=9700, verbose=True, **kwargs):
    """
    Convert from one given filter (in kwargs) to a target_filter

    Parameters
    ----------
    target_filter : str
       Target filter for the conversion
    temperature : float
       Temperature of star
    temperature_ref : int, optional
       Temperature of magnitude's zero-point
    verbose: boolean, optional
       Write out the conversion
    kwargs: dict
       It should have one element: filter=value, as the original filter.
       Any dot (.) in filter will be converted to '_' before passing it to
       get_filter.

    Returns
    -------
    float
       Converted value assuming a blackbody

    """

    if len(kwargs) != 1:
        raise ValueError("One, and only one reference filter needs to "
                         "be specified as kwargs ({0:s})".format(kwargs,))

    orig_filter, orig_value = list(kwargs.items())[0]
    orig_filter = getfilter(orig_filter.replace('_', '.'), True)
    target_filter = getfilter(target_filter.replace('_', '.'), True)

    if verbose:
        print(f"Converting from '{orig_filter}'={orig_value} to '{target_filter}'\n"
              f"(T={temperature} object)")

    ref0 = orig_value + 2.5 * np.log(applyfilter(orig_filter, temperature) /
                                     applyfilter(orig_filter, temperature_ref)) / np.log(10)

    return -2.5 * (np.log(applyfilter(target_filter, temperature) /
                          applyfilter(target_filter, temperature_ref)) /
                   np.log(10) + ref0)


def planeteff(au=1.0, tstar=6000, rstar=1.0, albedo=0.0):
    """
    Computes effective temperature of a planet based on the given
    parameters

    Parameters
    ----------
    au : float, optional
    tstar : int, optional
    rstar : float, optional
    albedo : float, optional

    Returns
    -------
    float
    """
    return tstar * np.sqrt((rstar * c.R_sun / au / c.au) * np.sqrt(1 - albedo) / 2.0)
