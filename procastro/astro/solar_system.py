import re
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcol
import matplotlib.path as mpath
import cartopy.crs as ccrs
from PIL import Image

from astropy import time as apt, units as u, coordinates as apc
from astropy.table import Table, QTable
from bs4 import BeautifulSoup

from procastro.astro.projection import new_x_axis_at, unit_vector, current_x_axis_to
from procastro.core.cache import jpl_cache, usgs_map_cache
import procastro as pa

TwoValues = tuple[float, float]


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
    for spec in specifications.split("\n"):
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
    """Read JPL's Horizons ephemeris file returning the adequate datatype in an astropy.Table with named columns

    Parameters
    ----------
    specification: str
       Ephemeris specification. It can be the filename to be read with the ephemeris, or with newline-separated
        commands for horizon's batch mode. It can also be a newline-separated string with the specifications
    Filename of the ephemeris file
    """
    if isinstance(specification, dict):
        specification = f"""!$$SOF\n{"\n".join([f'{k}={v}' for k, v in specification.items()])}"""

    specification = specification.strip()
    if specification.count("\n") == 0:  # filename is given
        filename = Path(specification)
        if not filename.exists():
            raise FileNotFoundError(f"File '{filename}' does not exists")
        with open(filename, 'r') as fp:
            line = fp.readline()
            if line[:6] == r"!$$SOF":
                ephemeris = _request_horizons_online(fp.read())
            else:
                ephemeris = open(specification, 'r').readlines()
    else:
        if specification[:6] != r"!$$SOF":
            raise ValueError(f"Multiline Horizons specification invalid:"
                             f"{specification}")
        ephemeris = _request_horizons_online(specification)

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
        sign = np.sign(coords.decd)
        sign += sign == 0
        df[f'dec_{name}'] = (coords.decd.abs() + (coords.decm + coords.decs / 60) / 60) * sign

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
        new_date = df[column].str.strip().str.replace(" ", "T")
        for idx, month in enumerate(['Jan', "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
            new_date = new_date.str.strip().str.replace(month, f"{idx+1:02d}")
        df[jd_col] = [apt.Time(ut_date).jd for ut_date in new_date]

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
              'mars': 499,
              'jupiter': 599,
              'saturn': 699,
              'uranus': 799,
              'neptune': 899,
              }
    match body:
        case int():
            pass
        case str():
            body = bodies[body.lower()]
        case a:
            raise ValueError(f"Invalid value in body ({a})")

    return {'COMMAND': body}


def jpl_times_from_time(times: str | apt.Time):
    """
Return dict with correct time batch call for JPL's horizon
    Parameters
    ----------
    times

    Returns
    -------

    """
    if isinstance(times, str):
        times = apt.Time(times, format='isot', scale='utc')
    if times.isscalar:
        times = apt.Time([times])
    times_str = " ".join([f"'{s}'" for s in times.jd])
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
             locations: list[TwoValues] | dict[str, TwoValues] = None,
             ax=None,
             show_poles="blue",
             detail=None,
             color="red", color_phase='black',
             color_background='black', color_title='white',
             ):
    """

    Parameters
    ----------
    color_title
       Color of title
    color_background
       Color of background
    color_phase
       Color of phase shading, None to skip
    detail
       submap of body
    color
    show_poles
    body
    observer
    time
    locations:
       a list of (lon, lat) coordinates
    ax
    """
    site = apc.EarthLocation.of_site(observer)
    request = jpl_observer_from_location(site) | jpl_times_from_time(time) | jpl_body_from_str(body)
    table = read_jpl(request)
    sub_obs_lon = table['ObsSub_LON'].value[0]
    sub_obs_lat = table['ObsSub_LAT'].value[0]
    np_ang = table['NP_ang'].value[0]

    lunar_to_observer = new_x_axis_at(sub_obs_lon, sub_obs_lat, z_pole_angle=np_ang)

    image = usgs_map_image(body, detail=detail)

    orthographic_image = get_orthographic(image,
                                          sub_obs_lon,
                                          sub_obs_lat,
                                          show_poles=show_poles)

    rotated_image = orthographic_image.rotate(-np_ang,
                                              resample=Image.Resampling.BICUBIC,
                                              expand=False, fillcolor=(255, 255, 255))
    x_offset = 0.5
    y_offset = 0
    ny, nx = rotated_image.size
    yy, xx = np.mgrid[-ny/2 + y_offset: ny/2 + y_offset, -nx/2 + x_offset: nx/2 + x_offset]
    rr = np.sqrt(yy**2 + xx**2).flatten()
    rotated_image.putdata([item if r < nx/2 - 1 else (0, 0, 0, 0)
                           for r, item in zip(rr, rotated_image.convert("RGBA").getdata())])

    f, ax = pa.figaxes(ax)
    f.patch.set_facecolor(color_background)
    ax.set_facecolor(color_background)
    ax.imshow(rotated_image)
    ax.axis('off')

    ax.set_title(f"{body.capitalize()} on {time.isot[:16]}",
                 color=color_title,
                 )
    transform_norm_to_axes = (mtransforms.Affine2D().scale(0.5) +
                              mtransforms.Affine2D().translate(0.5, 0.5) +
                              ax.transAxes)

    if len(locations) > 0:
        print(f"Location offset from {body.capitalize()}'s center (Delta_RA, Delta_Dec) in arcsec:")

        if isinstance(locations, list):
            locations = {str(i): v for i, v in enumerate(locations)}
        max_len = max([len(str(k)) for k in locations.keys()])
        for name, location in locations.items():
            rot_xy = lunar_to_observer.apply(unit_vector(*location, degrees=True))[1:3]
            delta_ra, delta_dec = table['Ang_diam'].value[0]*rot_xy/2

            ax.plot(*rot_xy,
                    transform=transform_norm_to_axes,
                    marker='d', color=color,
                    zorder=10,
                    )
            ax.annotate(f"{str(name)}: $\\Delta\\alpha$ {delta_ra:+.0f}\", $\\Delta\\delta$ {delta_dec:.0f}\"",
                        rot_xy,
                        xycoords=transform_norm_to_axes,
                        color=color,
                        zorder=10,
                        )

            format_str = f"{{name:{max_len+1}s}} {{delta_ra:+10.2f}} {{delta_dec:+10.2f}}"
            print(format_str.format(name=str(name),
                                    delta_ra=delta_ra, delta_dec=delta_dec))
        print("")

    if show_poles:
        ax.plot([-1, 1], [0, 0],
                color='blue',
                transform=transform_norm_to_axes,
                zorder=9,
                )
        ax.plot([0, 0], [-1, 1],
                color='blue',
                transform=transform_norm_to_axes,
                zorder=9,
                )

        for lat_pole in [-90, 90]:
            pole = lunar_to_observer.apply(unit_vector(0, lat_pole, degrees=True))
            ax.plot(*pole[1:3],
                    transform=transform_norm_to_axes,
                    alpha=1 - 0.5*(pole[0] < 0),
                    marker='o', color='blue',
                    zorder=9,
                    )

    if color_phase:
        n_phase = 50
        equatorial_to_border = new_x_axis_at(0, 90)
        border_to_terminator = current_x_axis_to(table['SunSub_LON'].value[0],
                                                 table['SunSub_LAT'].value[0],
                                                 )

        print(f"Sub Solar angle/distances: {table['SN_ang'][0]}/{table['SN_dist'][0]}")
        terminator_rotation = lunar_to_observer*border_to_terminator*equatorial_to_border
        terminator = terminator_rotation.apply(unit_vector(np.linspace(0, 360, 50),
                                                           0)
                                               )

        visible_terminator = np.array([(np.arctan2(t[2], t[1]), t[1], t[2])
                                       for t in terminator if t[0] > 0])
        angles = visible_terminator[:, 0]

        # first, let's put all angles monotonous ascending or descending from first item
        delta_angles = angles[1:] - angles[:-1]
        ups = np.where(delta_angles < -1)[0]
        downs = np.where(delta_angles > 1)[0]
        for up in ups:
            angles[up + 1:] += 2 * np.pi
        for down in downs:
            angles[down + 1:] -= 2 * np.pi
        # then, wherever there is the jump, is where the angle shold start
        descending = angles[1] < angles[0]
        delta_angles = (angles[1:] - angles[:-1])
        angles[:np.argmax(np.absolute(delta_angles)) + 1] += (1 - 2 * descending) * 2 * np.pi

        visible_terminator[:, 0] = angles
        visible_sorted_terminator = sorted(visible_terminator, key=lambda x: x[0])
        sub_sun = lunar_to_observer.apply(unit_vector(table['SunSub_LON'].value[0],
                                                      table['SunSub_LAT'].value[0],
                                                      )
                                          )
        angle_low = visible_sorted_terminator[0][0]
        angle_top = visible_sorted_terminator[-1][0]
        if angle_top < angle_low:
            angle_low -= 2 * np.pi

        mid_from_low = (angle_low + angle_top) / 2
        dark_from_low = (np.array([0, np.cos(mid_from_low), np.sin(mid_from_low)]) * sub_sun).sum() < 0

        if dark_from_low:
            angle_perimeter = np.linspace(angle_low, angle_top, n_phase)
            visible_sorted_terminator = visible_sorted_terminator[::-1]
        else:
            angle_perimeter = np.linspace(angle_top, angle_low + 2*np.pi, n_phase)
        perimeter = np.array([np.array([0, np.cos(ang), np.sin(ang)])
                              for ang in angle_perimeter]
                             + visible_sorted_terminator)

        clip_path = mpath.Path(vertices=perimeter[:, 1:3], closed=True)

        col = mcol.PathCollection([clip_path],
                                  facecolors=color_phase, alpha=0.7,
                                  edgecolors=(0, 0, 0, 0),
                                  zorder=4,
                                  transform=transform_norm_to_axes,
                                  )

        ax.add_collection(col)

        plt.plot(*sub_sun[1:3],
                 marker='d', color='yellow',
                 alpha=1 - 0.5 * (sub_sun[0] < 0),
                 transform=transform_norm_to_axes)

    return ax


def get_orthographic(platecarree_image,
                     sub_obs_lon,
                     sub_obs_lat,
                     show_poles="blue",
                     ):
    """
Returns ortographic projection with the specified center
    Parameters
    ----------
    platecarree_image
    sub_obs_lon
    sub_obs_lat
    show_poles
       Color to mark poles, set to "" if don't want to mark

    Returns
    -------

    """
    projection = ccrs.Orthographic(sub_obs_lon,
                                   sub_obs_lat)

    f = plt.figure(figsize=(3, 3))
    tmp_ax = f.add_axes((0, 0, 1, 1),
                        transform=f.transFigure,
                        projection=projection)

    tmp_ax.imshow(platecarree_image,
                  origin='upper',
                  transform=ccrs.PlateCarree(),
                  extent=(-180, 180, -90, 90),
                  )
    if show_poles:
        tmp_ax.plot(0, 90,
                    transform=ccrs.PlateCarree(),
                    marker='d', color=show_poles)
        tmp_ax.plot(0, -90,
                    transform=ccrs.PlateCarree(),
                    marker='d', color=show_poles)

    tmp_ax.axis('off')
    tmp_ax.set_global()  # the whole globe limits
    f.canvas.draw()

    image_flat = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)

    orthographic_image = image_flat.reshape(*reversed(f.canvas.get_width_height()), 3)
    orthographic_image = Image.fromarray(orthographic_image, 'RGB')
    plt.close(f)

    return orthographic_image


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


@usgs_map_cache
def usgs_map_image(body, detail=None, warn_multiple=True):
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def _parse_date(string):
        if string is None:
            return ""
        match len(string):
            case 8:
                return f"{string[:4]}-{month[int(string[4:6])-1]}{string[6:8]}"

        raise ValueError(f"Needs to implement parsing for date: {string}")

    directory = (Path(__file__).parents[0] / 'images')
    files = list(directory.glob("*.xml"))

    keywords = None
    if detail is not None:
        keywords = detail.split()

    # filter alternatives
    body_files = []
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            data = BeautifulSoup(f.read(), 'xml')
            body_in_xml = data.find("target").string
            if body.lower() == body_in_xml.lower():
                title = data.idinfo("title")[0].string
                if keywords is not None and not [k for k in keywords if k in title]:
                    continue

                info = [title,
                        file,
                        data.find("browsen").string,
                        data.idinfo("begdate")[0].string,
                        data.idinfo("enddate")[0].string,
                        ]
                if 'default' in str(file):
                    body_files.insert(0, info)
                else:
                    body_files.append(info)
    if len(files) == 0:
        detail_str = f" with detail keywords '{detail}'"
        raise ValueError(f"No map of '{body}' found{detail_str}")

    # select from alternatives
    if len(body_files) > 1:
        if warn_multiple:
            print("Several map alternatives were available (use space-separated keywords in 'detail' to filter).\n"
                  "Selected first of:")
            for bf in body_files:
                print(f"* {bf[0]} [{_parse_date(bf[3])}..{_parse_date(bf[4])}]")
            print("")

        body_files = [body_files[0]]

    # fetch alternative
    response = requests.get(body_files[0][2])
    return Image.open(BytesIO(response.content))
