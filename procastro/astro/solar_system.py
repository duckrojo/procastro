import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

import cartopy.crs as ccrs
from PIL import Image

from astropy import time as apt, units as u, coordinates as apc
from astropy.table import Table, QTable

from procastro.core.cache import jpl_cache
import procastro as pa


@jpl_cache
def _request_horizons_online(specifications):
    default_spec = {'MAKE_EPHEM': 'YES',
                    'EPHEM_TYPE': 'OBSERVER',
                    'CENTER': "'500@399'",
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
    custom_spec = {}
    prev = ""
    for spec in specifications:
        if spec[:6] == r"!$$SOF":
            continue
        kv = spec.strip().split("=")
        if len(kv) == 2:
            custom_spec[kv[0]] = kv[1]
            prev = kv[0]
        else:
            custom_spec[prev] += " " + kv[0]

    url = url_api + "&".join([f"{k}={v}" for k, v in (default_spec | custom_spec).items()])

    return eval(requests.get(url, allow_redirects=True).content)['result'].splitlines()


def read_jpl(specification):

    ephemeris = get_jpl_ephemeris(specification)
    return parse_jpl_ephemeris(ephemeris)


def get_jpl_ephemeris(specification):
    """Read JPL's Horizons ephemeris file returning the adequate datatype in a astropy.Table with named columns

    Parameters
    ----------
    specification: str
       Ephemeris specification. It can be the filename to be read with the ephemeris, or with newline-separated
        commands for horizon's batch mode. It can also be a newline-separated string with the specifications
    Filename of the ephemeris file
    """
    if isinstance(specification, dict):
        specification = f"""!$$SOF\n{"\n".join([f'{k}={v}' for k, v in specification.items()])}"""

    lines = specification.splitlines()

    if len(lines) == 1:  # filename is given
        filename = Path(specification)
        if not filename.exists():
            raise FileNotFoundError(f"File '{filename}' does not exists")
        with open(filename, 'r') as fp:
            line = fp.readline()
            if line[:6] == r"!$$SOF":
                ephemeris = _request_horizons_online(fp.readlines())
            else:
                ephemeris = open(specification, 'r').readlines()
    else:
        if lines[0][:6] != r"!$$SOF":
            raise ValueError(f"Multiline Horizons specification invalid:"
                             f"{lines}")
        ephemeris = _request_horizons_online(lines)

    return ephemeris


def parse_jpl_ephemeris(ephemeris):

    float_col = ['Date_________JDUT', 'APmag', 'S_brt',
                 'dRA_cosD', 'd_DEC__dt', 'dAZ_cosE', 'd_ELV__dt',
                 'SatPANG', 'a_mass', 'mag_ex',
                 'Illu_', 'Def_illu', 'Ang_diam',
                 'ObsSub_LON', 'ObsSub_LAT', 'SunSub_LON', 'SunSub_LAT',
                 'SN_ang', 'SN_dist', 'NP_ang', 'NP_dist', 'hEcl_Lon', 'hEcl_Lat',
                 'r', 'rdot', 'delta', 'deldot', 'one_way_down_LT', 'VmagSn', 'VmagOb',
                 'S_O_T', 'S_T_O', 'O_P_T', 'PsAng', 'PsAMV', 'PlAng',
                 'TDB_UT', 'ObsEcLon', 'ObsEcLat', 'N_Pole_RA', 'N_Pole_DC',
                 'GlxLon', 'GlxLat',  'Tru_Anom', 'phi',
                 'earth_ins_LT', 'RA_3sigma', 'DEC_3sigma', 'SMAA_3sig', 'SMIA_3sig', 'Theta Area_3sig',
                 'POS_3sigma', 'RNG_3sigma', 'RNGRT_3sig', 'DOP_S_3sig',  'DOP_X_3sig', 'RT_delay_3sig',
                 'PAB_LON', 'PAB_LAT', 'App_Lon_Sun',  'I_dRA_cosD', 'I_d_DEC__dt',
                 'Sky_motion', 'Sky_mot_PA', 'RelVel_ANG', 'Lun_Sky_Brt', 'sky_SNR',
                 'sat_primary_X', 'sat_primary_Y', 'a_app_Azi', 'a_app_Elev',
                 'ang_sep', 'T_O_M', 'MN_Illu_'
                 ]
    str_col = ['_slash_r', 'Cnst', 'L_Ap_Sid_Time', 'L_Ap_SOL_Time', 'L_Ap_Hour_Ang', ]
    convert_dict = {k: float for k in float_col} | {k: str for k in str_col}

    ut_col = ['Date___UT___HR_MN']
    jd_col = 'Date_________JDUT'

    coords_col = {'R_A_______ICRF______DEC': 'ICRF',
                  'R_A____a_apparent___DEC': 'apparent',
                  'RA__ICRF_a_apparnt__DEC': 'ICRF_app',
                  }
    sexagesimal_col = ['L_Ap_Sid_Time', 'L_Ap_SOL_Time', 'L_Ap_Hour_Ang',
                       ]
    two_values_col = ['X__sat_primary__Y', 'Azi_____a_app____Elev']
    slash_col = ['ang_sep_slash_v', 'T_O_M_slash_MN_Illu_']

    def change_names(string):
        string, _ = re.subn(r'1(\D)', r'one\1', string)
        string, _ = re.subn(r'399', 'earth', string)
        string, _ = re.subn(r'[%*().:-]', '_', string)
        string, _ = re.subn(r'/', '_slash_', string)
        return string

    previous = ""

    moon_presence = True
    while True:
        line = ephemeris.pop(0)
        if len(ephemeris) == 0:
            raise ValueError("No Ephemeris info: it should be surrounded by $$SOE and $$EOE")
        if re.match(r"\*+ *", line):
            continue
        if re.match(r"\$\$SOE", line):
            break
        if re.match(r'Center-site name', line):
            if 'GEOCENTRIC' in line:
                moon_presence = False
        previous = line

    # previous = change_names(previous)

    pattern = ""
    spaces = 0
    col_seps = re.split(r'( +)', previous.rstrip())

    if col_seps[0] == '':
        col_seps.pop(0)
    date = col_seps.pop(0)
    chars = len(date)
    pattern += f'(?P<{change_names(date)}>.{{{chars}}})'
    if moon_presence:
        pattern += f'(?P<moon_presence>.{{3}})'
        spaces = len(col_seps.pop(0)) - 3

    for val in col_seps:
        if val[0] == '\0':
            break
        elif val[0] != ' ':
            chars = len(val) + spaces
            pattern += f'(?P<{change_names(val)}>.{{{chars}}})'
        else:
            spaces = len(val)

    incoming = []
    while True:
        line = ephemeris.pop(0)
        if len(ephemeris) == 0:
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

    for column in sexagesimal_col:
        if column not in df:
            continue
        sexagesimal = pd.DataFrame()
        sexagesimal[['h', 'm', 's']] = df[column].str.strip().str.split(re.compile(" +"), expand=True)
        sexagesimal = sexagesimal.astype(float)
        df[f'{column}_float'] = sexagesimal.h + (sexagesimal.m + sexagesimal.s / 60) / 60

    # convert UT date to JD
    for column in ut_col:
        if column not in df:
            continue
        newdate = df[column].str.strip().str.replace(" ", "T")
        for idx, month in enumerate(['Jan', "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
            newdate = newdate.str.strip().str.replace(month, f"{idx+1:02d}")
        df[jd_col] = [apt.Time(ut_date).jd for ut_date in newdate]

    df['jd'] = df[jd_col]

    return Table.from_pandas(df.astype({k: v for k, v in convert_dict.items() if k in df}))


def path_from_jpl(body,
                  observer,
                  times: apt.Time,
                  ):
    """Get values sorted by jd on movement of Solar System Body

    Parameters
    ----------
    body
    observer
    times

    Returns
    -------
    object
    """

    time_spec = jpl_times_from_time(times)
    site = apc.EarthLocation.of_site(observer)

    request = jpl_body_from_str(body) | time_spec | jpl_observer_from_location(site)
    ret = read_jpl(request)

    ret['skycoord'] = apc.SkyCoord(ret['ra_ICRF'], ret['dec_ICRF'], unit=(u.hourangle, u.degree))

    return ret


def jpl_body_from_str(body):
    bodies = {'mercury': 199,
              'venus': 299,
              'moon': 301,
              'luna': 301,
              'mars': 399,
              'jupiter': 499,
              'saturn': 599,
              'uranus': 699,
              'neptune': 799,
              }
    match body:
        case int():
            pass
        case str():
            body = bodies[body.lower()]
        case a:
            raise ValueError(f"Invalid value in body ({a})")

    return {'COMMAND': body}


def jpl_times_from_time(times):
    if times.isscalar:
        times = apt.Time([times])
    times_str = "\n".join([f"'{s}'" for s in times.jd])
    return {'TLIST_TYPE': 'JD',
            'TIME_TYPE': 'UT',
            'TLIST': times_str}


def jpl_observer_from_location(site):
    return {'CENTER': "'coord@399'",
            'COORD_TYPE': "'GEODETIC'",
            'SITE_COORD': (f"'{site.lon.degree:.2f}, {site.lat.degree:.2f}," +
                           f" {site.height.to(u.km).value:.2f}'"),
            }


def body_map(body,
             observer,
             time: apt.Time,
             ax=None,
             ):
    site = apc.EarthLocation.of_site(observer)
    request = jpl_observer_from_location(site) | jpl_times_from_time(time) | jpl_body_from_str(body)
    table = read_jpl(request)
    projection = ccrs.Orthographic(table['ObsSub_LON'].value[0],
                                   table['ObsSub_LAT'].value[0])

    f, ax = pa.figaxes(10)
    pa.change_axes_projection(ax, projection)
    image = plt.imread(f'{os.path.dirname(__file__)}/images/{body}.jpg')
    ax.imshow(image,
              origin='lower',
              transform=ccrs.PlateCarree(),
              extent=[-180, 180, -90, 90],
              )

    ax.axis('off')
    ax.set_global()  # the whole globe limits
    f.canvas.draw()

    image_flat = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)

    image = image_flat.reshape(*reversed(f.canvas.get_width_height()), 3)
    image = Image.fromarray(image, 'RGB')

    image = image.rotate(table['NP_ang'],
                         resample=Image.Resampling.BICUBIC,
                         expand=False, fillcolor=(255, 255, 255))
    # image = white_transparent(image)

    ax.imshow(image)


def path_body(body,
              observer,
              times: apt.Time,
              use_jpl: bool = False,
              ):
    """

    Parameters
    ----------
    use_jpl
    body
    observer
    times
    """

    if use_jpl:
        return path_from_jpl(body, observer, times)

    site = apc.EarthLocation.of_site(observer)
    body_object = apc.get_body(body, times, location=site)

    return QTable([times, times.jd, body_object,
                   body_object.ra.degree, body_object.dec.degree],
                  names=['time', 'jd', 'skycoord', 'ra', 'dec'])
