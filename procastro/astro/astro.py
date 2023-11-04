#
#
# Copyright (C) 2014,2018 Patricio Rojo
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


__all__ = ['read_horizons_cols', 'get_transit_ephemeris',
           'blackbody', 'getfilter', 'applyfilter',
           'filter_conversion', 'planeteff',
           'find_target', 'moon_distance',
           'read_jpl', 'hour_angle_for_altitude', 'find_time_for_altitude',
           ]

from pathlib import Path
from typing import Union

import astropy.constants as c
import astropy.coordinates as apc
import astropy.units as u
import astropy.time as apt
import numpy as np
import glob
import os.path as path

import pandas as pd
import requests

import procastro as pa
from collections import defaultdict
import re
import warnings


try:
    import astroquery.simbad as aqs
except ImportError:
    aqs = None


def _request_horizons_online(specifications):
    default_spec = {'MAKE_EPHEM': 'YES',
                    'EPHEM_TYPE': 'OBSERVER',
                    'CENTER': "'500@399'",
                    'STEP_SIZE': "'2 DAYS'",
                    'QUANTITIES': "'1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,"
                                  "27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48'",
                    'REF_SYSTEM': "'ICRF'",
                    'CAL_FORMAT': "'JD'",
                    'CAL_TYPE': "'M'",
                    'TIME_DIGITS': "'MINUTES'",
                    'ANG_FORMAT': "'HMS'",
                    'APPARENT': "'AIRLESS'",
                    'RANGE_UNITS': "'AU'",
                    'SUPPRESS_RANGE_RATE': "'NO'",
                    'SKIP_DAYLT': "'NO'",
                    'SOLAR_ELONG': "'0,180'",
                    'EXTRA_PREC': "'NO'",
                    'R_T_S_ONLY': "'NO'",
                    'CSV_FORMAT': "'NO'",
                    'OBJ_DATA': "'YES'",
                    }
    url_api = "https://ssd.jpl.nasa.gov/api/horizons.api?"
    custom_spec = {spec.split("=")[0].replace(" ", ""): spec.strip().split("=")[1]
                   for spec in specifications
                   if spec[:6] != r"!$$SOF"}

    url = url_api + "&".join([f"{k}={v}" for k, v in (default_spec | custom_spec).items()])
    return eval(requests.get(url, allow_redirects=True).content)['result'].splitlines()


def read_jpl(filename):
    """Read JPL's Horizons ephemeris file returning the adequate datatype in a pandas dataframe with named columns

    Parameters
    ----------
    filename: str
    Filename of the ephemeris file
    """

    def change_names(string):
        string, _ = re.subn(r'1(\D)', r'one\1', string)
        string, _ = re.subn(r'399', 'earth', string)
        string, _ = re.subn(r'[%*().:-]', '_', string)
        string, _ = re.subn(r'/', '_slash_', string)
        return string

    float_col = ['Date_________JDUT', 'APmag', 'S_brt',
                 'dRA_cosD', 'd_DEC__dt', 'dAZ_cosE', 'd_ELV__dt',
                 'SatPANG', 'L_Ap_Sid_Time', 'a_mass mag_ex',
                 'Illu_', 'Def_illu', 'Ang_diam',
                 'ObsSub_LON', 'ObsSub_LAT', 'SunSub_LON', 'SunSub_LAT',
                 'SN_ang', 'SN_dist', 'NP_ang', 'NP_dist', 'hEcl_Lon', 'hEcl_Lat',
                 'r', 'rdot', 'delta', 'deldot', 'one_way_down_LT', 'VmagSn', 'VmagOb',
                 'S_O_T', 'S_T_O', 'O_P_T', 'PsAng', 'PsAMV', 'PlAng',
                 'TDB_UT', 'ObsEcLon', 'ObsEcLat', 'N_Pole_RA', 'N_Pole_DC',
                 'GlxLon', 'GlxLat',  'L_Ap_SOL_Time', 'Tru_Anom', 'L_Ap_Hour_Ang', 'phi',
                 'earth_ins_LT', 'RA_3sigma', 'DEC_3sigma', 'SMAA_3sig', 'SMIA_3sig', 'Theta Area_3sig',
                 'POS_3sigma', 'RNG_3sigma', 'RNGRT_3sig', 'DOP_S_3sig',  'DOP_X_3sig', 'RT_delay_3sig',
                 'PAB_LON', 'PAB_LAT', 'App_Lon_Sun',  'I_dRA_cosD', 'I_d_DEC__dt',
                 'Sky_motion', 'Sky_mot_PA', 'RelVel_ANG', 'Lun_Sky_Brt', 'sky_SNR',
                 'sat_primary_X', 'sat_primary_Y', 'a_app_Azi', 'a_app_Elev',
                 'ang_sep', 'T_O_M', 'MN_Illu_'
                 ]
    str_col = ['_slash_r', 'Cnst', ]
    ut_col = ['Date___UT___HR_MN']
    jd_col = 'Date_________JDUT'

    convert_dict = {k: float for k in float_col} | {k: str for k in str_col}

    coords_col = {'R_A_______ICRF______DEC': 'ICRF',
                  'R_A____a_apparent___DEC': 'apparent',
                  'RA__ICRF_a_apparnt__DEC': 'ICRF_app',
                  }
    two_values_col = ['X__sat_primary__Y', 'Azi_____a_app____Elev']
    slash_col = ['ang_sep_slash_v', 'T_O_M_slash_MN_Illu_']

    lines = filename.splitlines()
    if len(lines) == 1:
        if not Path(filename).exists():
            raise FileNotFoundError(f"File '{filename}' does not exists")
        with open(filename, 'r') as fp:
            line = fp.readline()
            if line[:6] == r"!$$SOF":
                filename = _request_horizons_online(fp.readlines())
    else:
        if lines[0][:6] != r"!$$SOF":
            raise ValueError(f"Multiline Horizons specification invalid:"
                             f"{lines}")
        filename = _request_horizons_online(lines)

    previous = ""

    if isinstance(filename, list):
        lines = filename
    elif Path(filename).exists():
        lines = open(filename, 'r').readlines()
    else:
        raise ValueError(f"Invalid specification of input: {filename}")

    while True:
        line = lines.pop(0)
        if len(lines) == 0:
            raise ValueError("No Ephemeris info: it should be surrounded by $$SOE and $$EOE")
        if re.match(r"\*+ *", line):
            continue
        if re.match(r"\$\$SOE", line):
            break
        previous = line

    # previous = change_names(previous)

    pattern = ""
    spaces = 0
    col_seps = re.split(r'( +)', previous.rstrip())
    if col_seps[0] == '':
        col_seps.pop(0)
    for val in col_seps:
        if val[0] != ' ':
            chars = len(val) + spaces
            pattern += f'(?P<{change_names(val)}>.{{{chars}}})'
        else:
            spaces = len(val)

    incoming = []
    while True:
        line = lines.pop(0)
        if len(lines) == 0:
            raise ValueError("No Ephemeris info: it should be surrounded by $$SOE and $$EOE")

        if re.match(r"\$\$EOE", line.rstrip()):
            break
        incoming.append(line.rstrip())

    # dataframe with each field separated
    df = pd.DataFrame({'incoming': incoming}).incoming.str.extract(pattern)
    df = df.replace(re.compile(r" +n\.a\."), np.nan).astype(str)

    # convert sexagesimal coordinates to float
    for coord, name in coords_col.items():
        if coord not in df:
            continue
        coords = pd.DataFrame()
        coords[['rah', 'ram', 'ras', 'decd', 'decm', 'decs']] = df[coord].str.strip().str.split(re.compile(" +"),
                                                                                                expand=True)
        coords = coords.astype(float)
        df[f'ra_{name}'] = coords.rah + (coords.ram + coords.ras / 60) / 60
        df[f'dec_{name}'] = (coords.decd.abs() + (coords.decm + coords.decs / 60) / 60) * np.sign(coords.decd)

    # convert two values into their own columns
    for column in two_values_col:
        if column not in df:
            continue
        local_pattern = re.compile(r"([a-zA-Z]+?)_+(.+?)_+([a-zA-Z]+?)$")
        match = re.match(local_pattern, column)
        left, right = f"{match[2]}_{match[1]}", f"{match[2]}_{match[3]}",
        new_df = df[column].str.strip().str.split(re.compile(" +"), n=2, expand=True)
        if len(new_df.columns) == 1:
            new_df[1] = new_df[0]
        df[[left, right]] = new_df

    # convert two values into their own columns
    for column in slash_col:
        if column not in df:
            continue
        local_pattern = re.compile(r"(.+)_slash_(.+)$")
        match = re.match(local_pattern, column)
        left, right = match[1], match[2]
        df[[left, right]] = df[column].str.split('/', expand=True)

    # convert UT date to JD
    for column in ut_col:
        newdate = df[column].str.strip().str.replace(" ", "T")
        for idx, month in enumerate(['Jan', "Feb", "Mar", "Apr", "May", "Jun",
                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
            newdate = newdate.str.strip().str.replace(month, f"{idx+1:02d}")
        df[jd_col] = [apt.Time(ut_date).jd for ut_date in newdate]

    df['jd'] = df[jd_col]

    return df.astype({k: v for k, v in convert_dict.items() if k in df})


def moon_distance(target, location=None, time=None):
    """Returns the distance of moon to target

    Parameters
    -------------
    target: str
        Target object for Moon distance
    location: apcoo.EarthLocation
        If None uses CTIO observatory
    time: apc.Time
        If None uses now().
    """

    target = find_target(target)
    if location is None:
        location = "ctio"
    if not isinstance(location, apc.EarthLocation):
        location = apc.EarthLocation.of_site(location)

    if time is None:
        time = apt.Time.now()
    if not isinstance(time, apt.Time):
        time = apt.Time(time)

    return apc.get_moon(time, location=location).separation(target)


def find_target(target, coo_files=None, equinox='J2000', extra_info=None, verbose=False):
    """
    Obtain coordinates from a target, that can be specified in various formats.

    Parameters
    ----------
    verbose
    extra_info
    target: str
       Either a coordinate understandable by astropy.coordinates
       (RA in hours, Dec in degrees), a name in coo_files, or a name
       resolvable by Simbad.
       Tests strictly in the previous order, returns as soon as it
       finds a match.
    coo_files: array_like, optional
       List of files that are searched for a match in target name.
       File should have at least three columns: Target_name RA Dec;
       optionally, a fourth column for comments. Target_name can have
       underscores that will be matched against spaces, dash, or no-character.
       Two underscores will additionally consider optional anything that
       follows (i.e. WASP_77__b, matches wasp-77, wasp77b, but not wasp77a).
       RA and Dec can be any mathematical expression that eval() can handle.
       RA is hms by default, unless 'd' is appended, Dec is always dms.
    equinox : str, optional
       Which astronomy equinox the coordinates refer. Default is J2000

    Returns
    -------
    SkyCoord object
       RA and Dec in hours and degrees, respectively.

    Raises
    ------
    ValueError
        If all query attempts fail (Wrong coordinates or unknown)
    """

    votable = {"sptype": "SP_TYPE"}
    if extra_info is None:
        extra_info = []

    try:
        ra_dec = apc.SkyCoord([f"{target}"], unit=(u.hour, u.degree),
                              equinox=equinox)
    except ValueError:
        if not isinstance(coo_files, (list, tuple)):
            coo_files = [coo_files]

        for coo_file in coo_files:
            if coo_file is None:
                continue
            # try:
            #     open_file = open(coo_file)
            # except TypeError:
            #     open_file=False
            try:
                open_file = open(coo_file)
            except IOError:
                pass
            else:
                with open_file:
                    for line in open_file.readlines():
                        if len(line) < 10 or line[0] == '#':
                            continue
                        name, ra, dec, note = line.split(None, 3)
                        extra = extra_info.copy()

                        for note_item in note.split():
                            if note_item.count("=") == 1:
                                key, val = note_item.split("=")
                                try:
                                    extra[extra_info.index(key)] = eval(val)
                                except ValueError:
                                    print("ignoring extra info not requested: {key}")
                        if ra[-1] == 'd':
                            ra = "{0:f}".format(float(ra[:-1]) / 15,)
                        if pa.accept_object_name(name, target):
                            if verbose:
                                print(f"Found in coordinate file: {coo_file}")
                            break
                    # this is to break out of two for loops as it should
                    # stop looking in other files
                    else:
                        continue
                    break
        # if coordinate not in file
        else:
            extra = []
            if verbose:
                print(" '{0:s}' not understood as coordinates, attempting query "
                      "as name... ".format(target,), end='')
            if aqs is None:
                raise ValueError(
                    "Sorry, AstroQuery not available for coordinate querying")

            custom_simbad = aqs.Simbad()
            if len(extra_info) > 0:
                for info in extra_info:
                    custom_simbad.add_votable_fields(info)

            query = custom_simbad.query_object(target)
            if query is None:
                # todo: make a nicer planet filtering option
                if target[-2] == ' ' and target[-1] in 'bcdef':
                    query = custom_simbad.query_object(target[:-2])

            if query is None:
                raise ValueError(
                    f"Target '{target}' not found on Simbad")
            ra, dec = query['RA'][0], query['DEC'][0]
            if len(extra_info) > 0:
                for info in extra_info:
                    if info in votable:
                        info = votable[info]
                    info = info.replace("(", "_")
                    info = info.replace(")", "")
                    extra.append(query[info.upper()][0])

        ra_dec = apc.SkyCoord('{0:s} {1:s}'.format(ra, dec),
                              unit=(u.hour, u.degree),
                              equinox=equinox)
        if verbose:
            print("success! \n  {})".format(ra_dec,))

    if len(extra_info) > 0:
        if len(extra_info) == len(extra):
            return ra_dec, extra
        else:
            print(f"Extra Info ({extra_info}) was not found: {extra}")
            return ra_dec, extra_info

    return ra_dec


def get_transit_ephemeris(target):
    """
    Recovers epoch, period and length of a target transit if said transit has
    been specified in one of the provided paths

    Transit files must be named ".transits" or "transits.txt", each transit
    should have the following columns separated by a space:

    {object_name} E{transit_epoch} P{transit_period} L{transit_length}

    If the object name contain spaces, replace them with an underscore when
    writing it into the file. On the other hand querying a name with spaces
    requires using spaces.

    An optional comment column can be placed at the end of a row placing a-mass
    'C' as prefix.

    Parameters
    ----------
    target: str
        Target requested

    Returns
    -------
    tr_epoch : float or None
    tr_period : float or None
    tr_length : float or None

    Raises
    ------
    ValueError
        If a data field does not match the specified format
    """
    paths = [pa.user_confdir("transits.txt")
             ]

    tr_epoch = None
    tr_period = None
    tr_length = None
    for transit_filename in paths:
        try:
            open_file = open(transit_filename)

            override = []
            print(transit_filename)
            for line in open_file.readlines():
                if line[0] == '#' or len(line) < 3:
                    continue
                data = line[:-1].split()
                planet = data.pop(0)

                if pa.accept_object_name(planet, target, planet_match=True):
                    for d in data:
                        if d[0].lower() == 'p':
                            override.append('period')
                            tr_period = float(eval(d[1:]))
                        elif d[0].lower() == 'e':
                            override.append("epoch")
                            tr_epoch = float(eval(d[1:]))
                        elif d[0].lower() == 'l':
                            override.append("length")
                            tr_length = float(eval(d[1:]))
                        elif d[0].lower() == 'c':
                            override.append("comment")
                        else:
                            raise ValueError("data field not understood, it "
                                             "must start with L, P, C, "
                                             "or E:\n{0:s}".format(line,))
                    print(f"Overriding for '{planet}' from file '{transit_filename}': {', '.join(override)}")

                if len(override):
                    break

        except IOError:
            pass

    return tr_epoch, tr_period, tr_length

###############################################################################
# Unused methods
######


def read_horizons_cols(file):
    """
    Reads a horizons ephemeris file into a dictionary. It maintains the
    original column's header as dictionary keys.
    Columns can be automatically recognized as float or int (work in progress
    the latter)

    Parameters
    ----------
    file : str

    Returns
    -------
    dict
    """

    mapping = defaultdict(lambda: str)
    map_float = ['dRA*cosD', 'd(DEC)/dt', 'a-mass', 'mag_ex', 'APmag',
                 'S-brt', 'Illu%', 'Ang-diam', 'Obsrv-lon', 'Obsrv-lat',
                 'Ob-lon', 'Ob-lat', 'SN.ang', 'SN.dist', 'delta', 'deldot',
                 'S-O-T', 'S-T-O', 'Solar-lon', 'Solar-lat', 'NP.ang',
                 'NP.dist', 'N.Pole-RA', 'N.Pole_Dec']
    map_int = []

    for f in map_float:
        mapping[f] = float
    for f in map_int:
        mapping[f] = int

    with open(file) as infile:
        data = defaultdict(list)
        pre_line = ''
        for line in infile:
            if line.startswith('$$SOE'):
                # FIXME: pre2_line is undefined
                quantities_ws = re.findall(r" +\S+", pre2_line)
                quantities_ns = [s.strip() for s in quantities_ws]
                break
            pre2_line = pre_line
            pre_line = line

        for line in infile:
            if line.startswith('>') or len(line) < 5:
                continue
            if line.startswith('$$EOE'):
                break

            cum = 0
            for qs, q in zip(quantities_ws, quantities_ns):
                item = mapping[q](line[cum:cum + len(qs)].strip())
                cum += len(qs)
                data[q].append(item)

        data = {d: np.array(data[d]) for d in data.keys()}

    return data


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


def hour_angle_for_altitude(dec, site_lat, altitude):
    """
    Returns hour angle at which the object reaches the requested altitude

    Parameters
    ----------
    dec
    site_lat
    altitude

    Returns
    -------
      Hour angle quantity,or 13 if the declination never reaches the altitude
    """
    cos_ha = (np.sin(altitude) - np.sin(dec) * np.sin(site_lat)
              ) / np.cos(dec) / np.cos(site_lat)
    mask = np.abs(cos_ha) > 1
    ret = (np.arccos(cos_ha)*u.radian).to(u.hourangle)
    ret[mask] = 13 * u.hourangle

    return ret


def find_time_for_altitude(location, time,
                           search_delta_hour: float = 2,
                           search_span_hour: float = 16,
                           fine_span_min: float = 20,
                           ref_altitude_deg: Union[str, float] = "min",
                           find: str = "next",
                           body: str = "sun"):
    """returns times at altitude with many parameters. The search span is centered around `time` and, by default,
     it searches half a day before and half a day after.

    Parameters
    ----------
    find: str
       find can be: 'next', 'previous'/'prev', or 'around'
    time: apt.Time
       starting time for the search. It must be within 4 hours of the middle of day to work with default parameters.
    ref_altitude_deg : float, str
       Altitude for which to compute the time. It can also be "min" or "max"
    """
    find_actions = {"next": 1,
                    "previous": -1,
                    "prev": -1,
                    "around": 1}
    multiplier = find_actions[find]

    rough_offset = - (find == 'around') * search_span_hour * u.hour / 2

    rough_span = time + np.arange(0, search_span_hour, search_delta_hour) * multiplier * u.hour + rough_offset

    altitude_rough = apc.get_body(body, rough_span,
                                location=location).transform_to(apc.AltAz(obstime=rough_span,
                                                                        location=location)
                                                                ).alt

    if isinstance(ref_altitude_deg, str):
        central_idx = getattr(np, f"arg{ref_altitude_deg}")(altitude_rough)
        ref_altitude = 0
        vertex = True
    else:
        ref_altitude = ref_altitude_deg * u.degree
        above = altitude_rough > ref_altitude
        central_idx = list(above).index(not above[0])
        vertex = False

    # following is number hours from time that has the requested elevation, roughly
    closest_idx = pa.parabolic_x(altitude_rough - ref_altitude, central_idx=central_idx, vertex=vertex) + central_idx
    closest_rough = closest_idx * search_delta_hour * multiplier * u.hour + rough_offset

    fine_span = time + closest_rough + np.arange(-fine_span_min, fine_span_min) * u.min

    sun = apc.get_body(body, fine_span)
    altitude = sun.transform_to(apc.AltAz(obstime=fine_span, location=location)).alt

    if isinstance(ref_altitude_deg, str):
        central_idx = getattr(np, f"arg{ref_altitude_deg}")(altitude)
        vertex = True
    else:
        central_idx = np.argmin(np.abs(altitude - ref_altitude))
        vertex = False

    # following is number hours from time that has the requested elevation, roughly
    closest_idx = pa.parabolic_x(altitude - ref_altitude,
                                 central_idx=central_idx,
                                 vertex=vertex) + central_idx

    if not (0 < closest_idx < len(altitude) - 1):
        if isinstance(ref_altitude_deg, str):
            label = f'{ref_altitude_deg} altitude'
        else:
            label = f'altitude {ref_altitude_deg} deg'
        newline = '\n'

        warnings.warn(f"It's possible that {label} was not found correctly "
                      f"{'after' if find else 'before'} {time} for body {body}.{newline}"
                      f"minimum index ({closest_idx}) on border: {altitude}{newline}"
                      f"But not quite what was expected from rough approx: {altitude_rough}")

    return time + (closest_idx - fine_span_min) * u.min + closest_rough
