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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

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
    df = df.replace(re.compile(r" +n\.a\."), np.nan).infer_objects(copy=False).astype(str)

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


def _body_map_video(body,
                    observer,
                    time: apt.Time,
                    ax: Axes | None = None,
                    color_background="black",
                    **kwargs,
                    ):

    backend = plt.get_backend()
    plt.switch_backend('agg')

    if ax is None:
        f, ax = pa.figaxes()
    else:
        f = ax.figure

    f.set_facecolor(color_background)
    title = ax.text(0, 1, "",
                    color='w',
                    transform=ax.transAxes, ha="left", va="top")

    def animate(itime):
        ax.clear()
        title.set_text(time[itime].isot)
        artists = body_map(body, observer, time[itime],
                           return_axes=False, verbose=False,
                           ax=ax,
                           **kwargs)
        return artists

    ani = FuncAnimation(f, animate, interval=40, blit=False, repeat=True, frames=len(time))

    filename = f"{body}_{time[0].isot.replace(":", "")[:17]}.gif"
    ani.save(filename,
             dpi=300, writer=PillowWriter(fps=25),
             )
    print(f"saved file {filename}")
    plt.switch_backend(backend)


def body_map(body,
             observer,
             time: apt.Time,
             locations: list[TwoValues] | dict[str, TwoValues] = None,
             detail=None, reread_usgs=False,
             radius_to_plot=None,
             ax: Axes | None = None, return_axes=True, verbose=True,
             color_location="red", color_phase='black',
             color_background='black', color_title='white',
             color_local_time='black', color_poles="blue",
             ):
    """

    Parameters
    ----------
    body:
       Body to show
    observer:
       Location of observer as understood by astropy.EarthLocation.of_site
    time:
       time in astropy.Time format. If passed a single element it returns an image, else a video
    locations:
       a list of (lon, lat) coordinates
    verbose:
       Whether to print coordinates and selected info
    reread_usgs:
       Reread USGS data even if cache exists if True. Use this when you want to get the full list of available maps
    return_axes:
       return axes if True, otherwise return artists
    radius_to_plot:
       this defines the limits to show in the axes in arcsec, if None then it will be set to the radius of the body.
    color_title
       Color of title.  None or blank to skip
    color_background
       Color of background
    color_phase
       Color of phase shading. None or blank to skip
    detail
       keywords to choose submap of body, None
    color_location
       color of the location marks
    color_local_time:
       color of the local timemarks. None or blank to skip
    color_poles
       color of the poles. None or blank to skip
    ax:
       matplotlib.Axes to use
    """

    if not time.isscalar:
        _body_map_video(body, observer, time,
                        locations=locations,
                        detail=detail, reread_usgs=reread_usgs,
                        radius_to_plot=radius_to_plot,
                        ax=ax,
                        color_poles=color_poles, color_local_time=color_local_time,
                        color_location=color_location, color_phase=color_phase,
                        color_background=color_background, color_title=color_title,
                        )
        return

    site = apc.EarthLocation.of_site(observer)
    request = jpl_observer_from_location(site) | jpl_times_from_time(time) | jpl_body_from_str(body)
    table = read_jpl(request)
    sub_obs_lon = table['ObsSub_LON'].value[0]
    sub_obs_lat = table['ObsSub_LAT'].value[0]
    np_ang = table['NP_ang'].value[0]

    lunar_to_observer = new_x_axis_at(sub_obs_lon, sub_obs_lat, z_pole_angle=-np_ang)
    body_to_local_time = new_x_axis_at(table['SunSub_LON'].value[0],
                                       table['SunSub_LAT'].value[0],
                                       )

    image = usgs_map_image(body, detail=detail, no_cache=reread_usgs)

    orthographic_image = get_orthographic(image,
                                          sub_obs_lon,
                                          sub_obs_lat,
                                          show_poles=color_poles)

    rotated_image = orthographic_image.rotate(np_ang,
                                              resample=Image.Resampling.BICUBIC,
                                              expand=False, fillcolor=(255, 255, 255))
    x_offset = 0.5
    y_offset = 0
    ny, nx = rotated_image.size
    yy, xx = np.mgrid[-ny/2 + y_offset: ny/2 + y_offset, -nx/2 + x_offset: nx/2 + x_offset]
    rr = np.sqrt(yy**2 + xx**2).flatten()

    color_background_rgb = (*[int(c*255) for c in to_rgba(color_background)[:3]], 0)
    rotated_image.putdata([item if r < nx/2 - 1 else color_background_rgb
                           for r, item in zip(rr, rotated_image.convert("RGBA").getdata())])

    f, ax = pa.figaxes(ax)
    f.patch.set_facecolor(color_background)
    ax.set_facecolor(color_background)
    ax.imshow(rotated_image,
              )
    ax.axis('off')

    if verbose:
        print(f"Sub Observer longitude/latitude: {table['ObsSub_LON'][0]}/{table['ObsSub_LAT'][0]}")
        print(f"Sub Solar longitude/latitude: {table['SunSub_LON'][0]}/{table['SunSub_LAT'][0]}")
        print(f"Sub Solar angle/distances: {table['SN_ang'][0]}/{table['SN_dist'][0]}")
        print(f"North Pole angle/distances: {table['NP_ang'][0]}/{table['NP_dist'][0]}")

    ax.set_facecolor(color_background)
    ang_rad = table['Ang_diam'].value[0]/2
    ax.imshow(rotated_image,
              extent=[-ang_rad, ang_rad, -ang_rad, ang_rad],
              )

    if radius_to_plot is None:
        radius_to_plot = ang_rad

    ax.set_xlim([-radius_to_plot, radius_to_plot])
    ax.set_ylim([-radius_to_plot, radius_to_plot])

    if color_title is not None and color_title:
        ax.set_title(f'{body.capitalize()} on {time.isot[:16]}. Radius {ang_rad:.1f}"',
                     color=color_title,
                     )
    transform_norm_to_axes = mtransforms.Affine2D().scale(ang_rad) + ax.transData

    if len(locations) > 0:
        if verbose:
            print(f"Location offset from {body.capitalize()}'s center (Delta_RA, Delta_Dec) in arcsec:")

        if isinstance(locations, list):
            locations = {str(i): v for i, v in enumerate(locations)}
        max_len = max([len(str(k)) for k in locations.keys()])
        for name, location in locations.items():
            position = unit_vector(*location, degrees=True)
            rot_xy = lunar_to_observer.apply(position)[1: 3]
            delta_ra, delta_dec = table['Ang_diam'].value[0]*rot_xy/2

            local_time_location = (np.arctan2(*body_to_local_time.apply(position)[:2][::-1]) + np.pi) * 12 / np.pi

            ax.plot(*rot_xy,
                    transform=transform_norm_to_axes,
                    marker='d', color=color_location,
                    alpha=1 - 0.5 * (rot_xy[0] < 0),
                    zorder=10,
                    )

            ax.annotate(f"{str(name)}: $\\Delta\\alpha$ {delta_ra:+.0f}\", "
                        f"$\\Delta\\delta$ {delta_dec:.0f}\"",
                        rot_xy,
                        xycoords=transform_norm_to_axes,
                        color=color_location,
                        alpha=1 - 0.5 * (rot_xy[0] < 0),
                        zorder=10,
                        )

            format_str = (f"{{name:{max_len+1}s}} {{delta_ra:+10.2f}} {{delta_dec:+10.2f}}"
                          f"   (LocalSolarTime: {{local_time_location:+7.2f}}h)")
            if verbose:
                print(format_str.format(name=str(name), local_time_location=local_time_location,
                                        delta_ra=delta_ra, delta_dec=delta_dec))
        if verbose:
            print("")

    if color_poles is not None and color_poles:
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

    if color_local_time:
        _add_local_time(ax,
                        (sub_obs_lon, sub_obs_lat),
                        (table['SunSub_LON'].value[0], table['SunSub_LAT'].value[0]),
                        np_ang,
                        color_phase,
                        transform_norm_to_axes,
                        )

    if color_phase is not None and color_phase:
        _add_phase_shadow(ax,
                          (sub_obs_lon, sub_obs_lat),
                          (table['SunSub_LON'].value[0], table['SunSub_LAT'].value[0]),
                          np_ang,
                          color_phase,
                          transform_norm_to_axes,
                          )

    if return_axes:
        return ax
    else:
        return ax.collections + ax.lines + ax.texts + ax.images


def _add_local_time(ax,
                    sub_obs,
                    sub_sun,
                    np_ang,
                    color,
                    transform_from_norm,
                    precision=50,
                    ):
    lunar_to_observer = new_x_axis_at(*sub_obs, z_pole_angle=-np_ang)
    local_time_to_body = current_x_axis_to(*sub_sun)

    artists = []
    for ltime in range(24):
        longitude_as_local_time = new_x_axis_at((12-ltime)*15, 0)

        local_time_rotation = lunar_to_observer * local_time_to_body * longitude_as_local_time
        local_time = local_time_rotation.apply(unit_vector(0, np.linspace(-90, 90, precision),
                                                           degrees=True)
                                               )
        visible = np.array([(y, z) for x, y, z in local_time if x > 0])
        n_visible = len(visible)
        if n_visible > 25:
            lines = ax.plot(visible[:, 0], visible[:, 1],
                            color=color,
                            alpha=0.7,
                            ls=':',
                            transform=transform_from_norm,
                            zorder=6,
                            linewidth=0.5,
                            )

            text = ax.annotate(f"{ltime}$^h$", (visible[n_visible // 2][0],
                                                visible[n_visible // 2][1]),
                               color=color,
                               xycoords=transform_from_norm,
                               alpha=0.7, ha='center', va='center',
                               )

            artists.extend([lines, text])

    return tuple(artists)


def get_orthographic(platecarree_image,
                     sub_obs_lon,
                     sub_obs_lat,
                     marks=None,
                     show_poles="",
                     ):
    """
Returns ortographic projection with the specified center
    Parameters
    ----------
    platecarree_image
    sub_obs_lon
    sub_obs_lat
    show_poles
       Color to mark poles, set to "" to skip

    Returns
    -------

    """
    projection = ccrs.Orthographic(sub_obs_lon,
                                   sub_obs_lat)

    backend = plt.get_backend()
    plt.switch_backend('agg')

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

    if marks is not None:
        for mark in marks:
            tmp_ax.plot(*mark,
                        transform=ccrs.PlateCarree(),
                        marker='x', color=show_poles
                        )

    tmp_ax.axis('off')
    tmp_ax.set_global()  # the whole globe limits
#    tmp_ax.gridlines()
    f.canvas.draw()

    image_flat = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)

    orthographic_image = image_flat.reshape(*reversed(f.canvas.get_width_height()), 3)
    orthographic_image = Image.fromarray(orthographic_image, 'RGB')
    plt.close(f)

    plt.switch_backend(backend)

    return orthographic_image


def body_path(body,
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
    """

    Parameters
    ----------
    body
    detail
    warn_multiple
    no_cache
       If True then it will reread options from USGS even if cache exists

    Returns
    -------

    """
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

    detail_str = f" with detail keywords '{detail}'"
    if len(body_files) == 0:
        raise ValueError(f"No map of '{body}' found{detail_str}")
    # select from alternatives
    elif len(body_files) > 1:
        if warn_multiple:
            suggest = f" (use space-separated keywords in 'detail' to filter).\n" if not detail_str else ""
            print(f"Several map alternatives for {body} were available{detail_str}\n{suggest}"                  
                  "Selected first of:")
            for bf in body_files:
                print(f"* {bf[0]} [{_parse_date(bf[3])}..{_parse_date(bf[4])}]")
            print("")

        body_files = [body_files[0]]

    # fetch alternative
    response = requests.get(body_files[0][2])
    return Image.open(BytesIO(response.content))


def _add_phase_shadow(ax,
                      sub_obs,
                      sub_sun,
                      np_ang,
                      color,
                      transform_from_normal,
                      precision=50):

    def vector_to_observer(vector):
        rotation = new_x_axis_at(*sub_obs, z_pole_angle=-np_ang)
        return rotation.apply(vector)

    upper_vector_terminator = np.cross(unit_vector(*sub_obs), unit_vector(*sub_sun))
    upper_shadow_from_sun = new_x_axis_at(*sub_sun).apply(upper_vector_terminator)
    upper_angle_from_sun = (np.arctan2(*upper_shadow_from_sun[1:][::-1]) - np.pi / 2) * 180 / np.pi

    terminator = unit_vector(-90, np.linspace(-90, 90, precision))
    visible_terminator_at_body = current_x_axis_to(*sub_sun, z_pole_angle=-upper_angle_from_sun).apply(terminator)
    visible_terminator = vector_to_observer(visible_terminator_at_body)

    angle_top, angle_low = np.arctan2(*visible_terminator[:, 1:][[-1, 0]].transpose()[::-1]
                                      )
    if angle_top > angle_low:
        angle_top -= 2 * np.pi

    angle_perimeter = np.linspace(angle_top, angle_low, precision)
    perimeter = np.array([np.array([0, np.cos(ang), np.sin(ang)])
                          for ang in angle_perimeter]
                         + list(visible_terminator))

    clip_path = mpath.Path(vertices=perimeter[:, 1:3], closed=True)

    col = mcol.PathCollection([clip_path],
                              facecolors=color, alpha=0.7,
                              edgecolors='none',
                              zorder=7,
                              transform=transform_from_normal,
                              )
    ax.add_collection(col)

    sub_sun_marker = ax.plot(*vector_to_observer(unit_vector(*sub_sun))[1:],
                             marker='d', color='yellow',
                             alpha=1 - 0.5 * (sub_sun[0] < 0),
                             transform=transform_from_normal)

    return col, sub_sun_marker

