import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy import time as apt, units as u, coordinates as apc
from astropy.table import Table, QTable

from procastro.core.cache import jpl_cache


@jpl_cache
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
    """Read JPL's Horizons ephemeris file returning the adequate datatype in a astropy.Table with named columns

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

    return Table.from_pandas(df.astype({k: v for k, v in convert_dict.items() if k in df}))


def path_from_jpl(body,
                  observer,
                  time_start: apt.Time,
                  delta_or_finish_time: apt.Time | apt.TimeDelta | u.Quantity,
                  steps: int | u.Quantity = 10,
                  ):
    """Get values sorted by jd on movement of Solar System Body

    Parameters
    ----------
    body
    observer
    time_start
    delta_or_finish_time
    steps

    Returns
    -------
    object
    """
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
    site = apc.EarthLocation.of_site(observer)
    match delta_or_finish_time:
        case apt.Time():
            delta = delta_or_finish_time - time_start
        case apt.TimeDelta() | u.Quantity():
            delta = apt.TimeDelta(delta_or_finish_time)
        case a:
            raise ValueError(f"Invalid value, only delta or finish"
                             f" time in astropy format is allowed in {a}")
    match steps:
        case int():
            pass
        case apt.TimeDelta() | u.Quantity():
            steps = (delta/steps).to(u.dimensionless_unscaled)
        case a:
            raise ValueError(f"Invalid value in steps ({a})")

    step_size = (delta / steps).to(u.day).value
    if step_size >= 2:
        unit = "DAYS"
    else:
        step_size *= 24
        if step_size >= 2:
            unit = "HOURS"
        else:
            step_size *= 60
            if step_size >= 2:
                unit = "MINUTES"
            else:
                step_size *= 60
                unit = "SECONDS"
                if step_size < 2:
                    step_size = 1

    match body:
        case int():
            pass
        case str():
            body = bodies[body.lower()]
        case a:
            raise ValueError(f"Invalid value in body ({a})")

    request = {'STEP_SIZE': f"'{step_size:.0f} {unit}'",
               'CENTER': (f"'g: {site.lon.degree:.2f}, {site.lat.degree:.2f}," +
                          f" {site.height.to(u.km).value:.2f} @ 399'"),
               'START_TIME': f"'{time_start.isot[:16].replace("T", " ")}'",
               'STOP_TIME': f"'{(time_start + delta).isot[:16].replace("T", " ")}'",
               'COMMAND': body,
               }

    request_str = f"""!$$SOF
        {"\n".join([f'{k}={v}' for k, v in request.items()])}"""

    return read_jpl(request_str)


def path_body(body,
              observer,
              time_start_or_all: apt.Time,
              delta_or_finish_time: apt.Time | apt.TimeDelta | u.Quantity | None = None,
              steps: int | u.Quantity = 10,
              use_jpl: bool = False,
              ):
    """

    Parameters
    ----------
    use_jpl
    body
    observer
    time_start_or_all
    delta_or_finish_time
    steps
    """
    if delta_or_finish_time is None:
        if time_start_or_all.isscalar:
            time_start_or_all = apt.Time([time_start_or_all])
    else:
        if time_start_or_all.isscalar:
            match delta_or_finish_time:
                case apt.Time():
                    time_start_or_all = time_start_or_all + np.linspace(0, 1, steps
                                                                        ) * (delta_or_finish_time -
                                                                             time_start_or_all)
                case apt.TimeDelta() | u.Quantity():
                    time_start_or_all = time_start_or_all + np.linspace(0, 1, steps
                                                                        ) * delta_or_finish_time
                case _:
                    raise ValueError(f"Invalid value in delta_or_finish_time ({delta_or_finish_time})")
        else:
            warnings.warn("Both multiple times and delta_time has been specified. Ignoring the latter.")

    if use_jpl:
        return path_from_jpl(body, observer, time_start_or_all[0],
                             time_start_or_all[1] - time_start_or_all[0],
                             steps)

    site = apc.EarthLocation.of_site(observer)
    body_object = apc.get_body(body, time_start_or_all, location=site)

    return QTable([time_start_or_all, time_start_or_all.jd, body_object,
                   body_object.ra.degree, body_object.dec.degree],
                  names=['time', 'jd', 'skycoord', 'ra', 'dec'])
