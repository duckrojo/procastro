import warnings
from typing import Sequence

import numpy as np
from astropy import coordinates as apc, time as apt, units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import Table
from astropy.units import UnitConversionError
from astroquery.simbad import Simbad

from matplotlib import pyplot as plt, ticker
import cartopy.crs as ccrs
from matplotlib.patches import Polygon

import procastro as pa
from procastro.api_provider.api_service import ApiService
from procastro.astro import aqs as aqs
from procastro.misc.general import accept_object_name

solar_system_ephemeris.set('jpl')

def starry_plot(star_table: list[Table] | Table,
                projection="Robinson",
                marker_size=(4, 16),
                areas: list[apc.SkyCoord] = None,
                areas_facecolor: list[str] | str = 'none',
                areas_edgecolor: list[str] | str = 'black',
                areas_zorder: list[int] | int = 1,
                stars_zorder: int = 10,
                stars_color: str = "blue",
                ax=None, frameon=True,
                ):

    def ra_formatter(x, pos):
        x /= 15
        h = int(x)
        x = (x % 1) * 60
        m = x

        return f"{h}$^h${m:02.0f}"

    def dec_formatter(x, pos):
        return f"{int(x):+.0f}$^\\circ${(x%1)*60:02.0f}"

    if projection is not None:
        transform_projection = ccrs.PlateCarree()
    else:
        transform_projection = None

    if not isinstance(star_table, Sequence):
        star_table = [star_table]
    if isinstance(stars_color, str):
        stars_color = [stars_color]

    for table, color in zip(star_table, stars_color):
        if len(table) == 0:
            continue
        mags = table['flux']
        mmin = min(mags)
        mmax = max(mags)
        dm = (marker_size[1] - marker_size[0]) / (mmax - mmin)

        if ax is None:
            f = plt.figure()
            if projection is not None:
                projection = getattr(ccrs, projection)()
            ax = f.add_subplot(111,
                               projection=projection,
                               frameon=frameon,
                               )
        if projection is not None:
            gl = ax.gridlines(draw_labels=True, alpha=0.9, linestyle=':')
            gl.xformatter = ticker.FuncFormatter(ra_formatter)
            gl.yformatter = ticker.FuncFormatter(dec_formatter)


        ax.scatter(table['ra'], table['dec'],
                   transform=transform_projection,
                   s=marker_size[0] + (mags - mmin) * dm,
                   marker='o',
                   zorder=stars_zorder,
                   color=color,
                   )

    if areas is not None:
        if isinstance(areas_facecolor, str):
            areas_facecolor = [areas_facecolor] * len(areas)
        if isinstance(areas_edgecolor, str):
            areas_edgecolor = [areas_edgecolor] * len(areas)
        if isinstance(areas_zorder, int):
            areas_zorder = [areas_zorder] * len(areas)

        for area, zorder, facecolor, edgecolor in zip(areas,
                                                      areas_zorder,
                                                      areas_facecolor,
                                                      areas_edgecolor):
            points = [(coo.ra.degree, coo.dec.degree) for coo in area]
            patch = Polygon(points,
                            transform=transform_projection,
                            zorder=zorder,
                            facecolor=facecolor,
                            edgecolor=edgecolor,
                            )
            ax.add_patch(patch)

    return ax


def moon_distance(target, location=None, obs_time=None):
    """Returns the distance of moon to target

    Parameters
    -------------
    target: str
        Target object for Moon distance
    location: apcoo.EarthLocation
        If None uses CTIO observatory
    obs_time: apc.Time
        If None uses now().
    """

    if not isinstance(target, apc.SkyCoord):
        target = find_target(target)

    if location is None:
        location = "ctio"
    if not isinstance(location, apc.EarthLocation):
        location = apc.EarthLocation.of_site(location)

    if obs_time is None:
        obs_time = apt.Time.now()
    if not isinstance(obs_time, apt.Time):
        obs_time = apt.Time(obs_time)

    return apc.get_body("moon", obs_time, location=location).separation(target)


def polygon_around_path(coordinates: apc.SkyCoord | Sequence,
                        radius: u.Quantity = 5 * u.arcmin,
                        n_points: int = 11,
                        close: bool = False,
                        ):
    def border_points(center: apc.SkyCoord,
                      pa_next: apc.angles.core.Angle | float,
                      separation: u.Quantity,
                      clockwise: bool = True,
                      ):
        """

        Parameters
        ----------
        center
        pa_next:
           Position angle of next point
        separation
        clockwise

        Returns
        -------

        """
        if not isinstance(pa_next, apc.angles.core.Angle):
            pa_next *= u.deg
        angles = np.linspace(pa_next + 90 * u.deg, pa_next + 270 * u.deg, n_points)
        if not clockwise:
            angles = angles[::-1]
        return [center.directional_offset_by(a, separation) for a in angles]

    def offset_three(coo1, coo2, coo3, separation):
        pa_12 = coo1.position_angle(coo2)
        pa_23 = coo2.position_angle(coo3)
        delta = pa_23 - pa_12
        delta += 360 * u.deg * (delta < 0 * u.deg)
        if delta < 180*u.deg:  # expand
            return [coo2.directional_offset_by(pa_12 - 90 * u.deg, separation),
                    coo2.directional_offset_by(pa_23 - 90 * u.deg, separation),
                    ]
        else:  # reduce
            return [coo2.directional_offset_by((pa_12 + pa_23)/2 - 90 * u.deg, separation),
                    ]

    if coordinates.isscalar:
        offsets = apc.SkyCoord([border_points(coordinates, 0, radius),
                                border_points(coordinates, 180, radius)[1:-1]])
    else:
        offsets = border_points(coordinates[0], coordinates[0].position_angle(coordinates[1]), radius)
        for idx in np.arange(len(coordinates)-2) + 1:
            offsets.extend(offset_three(coordinates[idx-1], coordinates[idx], coordinates[idx+1], radius))
        offsets.extend(border_points(coordinates[-1], coordinates[-2].position_angle(coordinates[1]), radius))
        for idx in (np.arange(len(coordinates)-2) + 1)[::-1]:
            offsets.extend(offset_three(coordinates[idx+1], coordinates[idx], coordinates[idx-1], radius))

    if close:
        offsets.append(offsets[0])

    return apc.SkyCoord([o.ra for o in offsets], [o.dec for o in offsets])


def simbad_around_path(path: apc.SkyCoord | Sequence,
                       radius: u.Quantity = 5 * u.arcmin,
                       exclude_radius: u.Quantity | None = None,
                       brightest: float = 5,
                       dimmest: float = 11,
                       filter_name: str = 'V',
                       points_hemisphere: int = 7,
                       ):

    include_polygon = polygon_around_path(path, radius, n_points=points_hemisphere)
    if exclude_radius is None:
        exclude_polygon = None
    else:
        exclude_polygon = polygon_around_path(path, exclude_radius, n_points=points_hemisphere)

    return simbad_between_polygons(include_polygon, exclude_polygon,
                                   brightest=brightest,
                                   dimmest=dimmest,
                                   filter_name=filter_name,
                                   )


def simbad_between_polygons(include_polygon: apc.SkyCoord | Sequence,
                            exclude_polygon: u.Quantity | None = None,
                            brightest: float = 5,
                            dimmest: float = 11,
                            filter_name: str = 'V',
                            decimals=6,
                            ):
    """

    Parameters
    ----------
    include_polygon
      polygon made of SkyCoord points inside of which to search for stars
    exclude_polygon
      polygon made of SkyCoord points inside of which to omit stars
    brightest
      brightest magnitude
    dimmest
      dimmest magnitude
    filter_name
      filter in which to look for magnitude
    decimals
      decimals of degrees used in specifying the polygons

    Returns
    -------
      astropy.Table with "name, ra, dec, skycoord, magnitude" columns

    """
    def adql_from_polygon(polygon, digits):
        if polygon is None:
            return ""
        coo_string = f", {{ra:.{digits}f}}, {{dec:.{digits}f}}"
        polygon_string = "".join([coo_string.format(ra=coo.ra.degree, dec=coo.dec.degree) for coo in polygon])
        ret = (" AND CONTAINS(POINT('ICRS', ra, dec),"
               " POLYGON('ICRS'{polygon_string}))".format(polygon_string=polygon_string))
        return ret

    include_string = adql_from_polygon(include_polygon, decimals)
    include_string += "= 1"
    exclude_string = adql_from_polygon(exclude_polygon, decimals)
    exclude_string += "= 0" * (exclude_string != "")

    query = ("SELECT main_id, ra, dec, flux.flux "
             "FROM basic JOIN flux ON basic.oid=flux.oidref "
             f"{include_string}"
             f"{exclude_string}"
             f" AND flux.filter='{filter_name}'"
             f" AND flux.flux>{brightest} AND flux.flux<{dimmest};"
             )

    return Simbad.query_tap(query)


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

                        # RA can be given in hours or degree if prepending a d
                        ra = float(ra[:-1]) / 15 if ra[-1] == 'd' else float(ra)
                        dec = float(dec)
                        if accept_object_name(name, target):
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


            apiService = ApiService(simbad_votable_fields=extra_info if len(extra_info) > 0 else None)
            query = apiService.request_simbad(object_name= target).data # accesing to the data by accessing the data parameter.
            if query is None:
                # todo: make a nicer planet filtering option
                if target[-2] == ' ' and target[-1] in 'bcdef':
                    query = apiService.request_simbad(object_name = target[:-2]).data

                    if not len(query):
                        query = custom_simbad.query_object(target[:-3])

            if not len(query):
                raise ValueError(
                    f"Target '{target}' not found on Simbad")
            ra = (query['ra'][0] * query['ra'].unit).to(u.hourangle).value
            dec = (query['dec'][0] * query['dec'].unit).to(u.deg).value
            if len(extra_info) > 0:
                for info in extra_info:
                    if info in votable:
                        info = votable[info]
                    info = info.replace("(", "_")
                    info = info.replace(")", "")
                    extra.append(query[info.upper()][0])

        ra_dec = apc.SkyCoord('{0:f} {1:f}'.format(ra, dec),
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


def hour_angle_for_altitude(dec: np.ndarray | float,
                            site_lat: np.ndarray | float,
                            altitude_deg: str | np.ndarray | float
                            ) -> np.ndarray | float:
    """
    Returns hour angle at which an object with the given coordinate reaches the requested altitude. The
    quantities can be either scalar or array, if the latter they must have all the same size

    Parameters
    ----------
    dec:
       Declination of target(s) in radian
    site_lat:
       Latititude of observation in radian
    altitude:
       Requested altitude in radian

    Return
    -------
      Hour angle quantity, or 13 for each target that does not reach the altitude.
    """
    if altitude_deg == "max":
        cos_ha = 1.0
    # return check_precision(transit_time_if_no_motion[use_past])
    elif altitude_deg == "min":
        cos_ha = -1.0
    else:
        cos_ha = (np.sin(altitude_deg * np.pi / 180) - np.sin(dec) * np.sin(site_lat)
              ) / np.cos(dec) / np.cos(site_lat)

    fail = np.abs(cos_ha) > 1
    if isinstance(fail, np.ndarray):
        if any(fail):
            raise ValueError(f"One of the objects with declination {dec} does not reach altitude {altitude_deg} from site")
    else:
        if fail:
            raise ValueError(f"Object with declination {dec} does not reach altitude {altitude_deg} from site")
    ret = (np.arccos(cos_ha)*u.radian).to(u.hourangle)

    return ret


def find_time_for_altitude(location, time,
                           ref_altitude_deg: str | float,
                           find: str = "next",
                           body: str | apc.SkyCoord = "sun",
                           mean_apparent="apparent", # whether to include nutation
                           verbose=False,
                           ):
    """returns times at altitude with many parameters. The search span is centered around `time` and, by default,
     it searches half a day before and half a day after.

    Parameters
    ----------
    location
    ref_altitude_deg : float, str
       Altitude for which to compute the time. It can also be "min" or "max"
    search_delta_hour
    search_span_hour
    fine_span_min
    body
    find: str
       find can be: 'next', 'previous'/'prev', or 'around/closer'
    time: apt.Time
       starting time for the search. It must be within 4 hours of the middle of day to work with default parameters.
    """

    ref_time = apt.Time(time)
    sidereal_to_solar = (u.sday / u.day).to(u.dimensionless_unscaled)
    lst_start = ref_time.sidereal_time(mean_apparent, location).to(u.hourangle).value
    if isinstance(body, apc.SkyCoord):
        body_start = body
    else:
        body_start = apc.get_body(body, ref_time, location)

    delta_ha = body_start.ra.to(u.hourangle).value - lst_start

    if verbose:
        print(f"********** finding {find}. LST {lst_start}")

    return_times = []
    for transit in [-1, 1]:
        ha_to_transit = transit * ((transit * delta_ha) % 24) * u.hour
        transit_time_approx = ref_time + ha_to_transit * sidereal_to_solar

        if verbose:
            print(f"APPROX. HA to transit: {ha_to_transit},  time: {transit_time_approx}")

        for before_after in [-1, 1]:
            hour_angle_elevation = hour_angle_for_altitude(body_start.dec.to(u.radian).value,
                                                           location.lat.to(u.radian).value,
                                                           ref_altitude_deg)
            elevation_time_approx = (transit_time_approx +
                                     before_after * hour_angle_elevation.to(u.hourangle).value *
                                     sidereal_to_solar * u.hour)
            if isinstance(body, apc.SkyCoord):
                body_at_approx = body
            else:
                body_at_approx = apc.get_body(body, elevation_time_approx, location)

            delta_ha_transit_improved = body_at_approx.ra.to(u.hourangle).value - lst_start
            ha_to_transit_improved = transit * ((transit * delta_ha_transit_improved) % 24) * u.hourangle
            ha_to_elevation_improved = hour_angle_for_altitude(body_at_approx.dec.to(u.radian).value,
                                                               location.lat.to(u.radian).value,
                                                               ref_altitude_deg)

            improved_transit_time = ref_time + ha_to_transit_improved.value * sidereal_to_solar * u.hour
            improved_elevation_time = (improved_transit_time +
                                       before_after * ha_to_elevation_improved.value * sidereal_to_solar * u.hour)

            return_times.append(improved_elevation_time)

            if verbose:
                if isinstance(body, apc.SkyCoord):
                    computed = body
                else:
                    computed = apc.get_body(body, improved_elevation_time, location)

                altaz = computed.transform_to(apc.AltAz(obstime=improved_elevation_time, location=location))
                print(f"APPROX. HA to elevation: {hour_angle_elevation},  time: {elevation_time_approx}")
                print(f"IMPROVED. HA to transit: {ha_to_transit_improved}. time: {improved_transit_time}")
                print(f"IMPROVED. HA to elevation: {ha_to_elevation_improved}. time {improved_elevation_time}")
                print(f"computed altitude: {altaz.alt}")
                print(f"IMPROVED ({transit}, {before_after}). ")


    return_times = apt.Time(return_times)
    time_delta = return_times - time

    idx_after = np.nonzero(time_delta > 0*u.day)[0][0]
    use_past = "prev" in find or (("around" in find or "closer" in find)
                                  and time_delta[idx_after] > abs(time_delta[idx_after-1]))
    return_time = return_times[idx_after - use_past]

    return return_time



if __name__ == "__main__":
    body = "sun"
    loc = apc.EarthLocation.of_site('lco')
    time = apt.Time.now() + 1*u.day
    time = apt.Time("2025-06-12 16:50:00.0")
    elev = 0

    sun = apc.get_body(body, time, loc)

    print(f"from time {time} looking for elevation {elev}")

    tt = find_time_for_altitude(loc, time, elev, body=body, find="next", verbose=True)
    print(tt)

    tt = find_time_for_altitude(loc, time, elev, body=body, find="prev", verbose=True)
    print(tt)

    tt = find_time_for_altitude(loc, time, elev, body=body, find="closer", verbose=True)
    print(tt)

    # from time 2025-06-11 3:21:52.438210 looking for elevation 15
    #  2025-Jun-11 03:22  m  256.493871 -70.380749  15 58 08.6767    0.0000 /?   2.4197683   86.055435   0.6136423
    #
    # starting from 2025-06-11 03:21:52.438210 (LST: 15:57:0.98, 15.966287524495621) and searching for next 15 deg
    # Hour Angle to transit: [-10.68287181  13.31712819], and then to requested: 3.739967715627403 hourangle
    # bracketing transit times: 2025-06-10T16:42:39.108 - 2025-06-11T16:38:43.198
    # found elev 14.407721624000189 deg at time 2025-06-11 12:54:56.076633 for next search


    # airless
    #I 2025-Jun-10 20:25 *   308.044180  15.286833    0.0000 /?   2.3698420   86.004757   0.9846484
    #C 2025-Jun-10 20:26 *   307.886433  15.114463    0.0000 /?   2.3699585   86.006010   0.9863962
    #R 2025-Jun-10 20:27 *   307.729248  14.941724    0.0000 /?   2.3700755   86.007259   0.9881320
    # 2025-Jun-10 20:28 *   307.572622  14.768619    0.0000 /?   2.3701928   86.008503   0.9898559
    # 2025-Jun-10 20:29 *   307.416552  14.595150    0.0000 /?   2.3703105   86.009743   0.9915679
    #
    #C 2025-Jun-11 12:54 *    52.832369  14.245390    0.0000 /?   2.3717287   85.765896   -0.247887
    # 2025-Jun-11 12:55 *    52.677285  14.419374    0.0000 /?   2.3716052   85.766707   -0.246160
    #I 2025-Jun-11 12:56 *    52.521651  14.592998    0.0000 /?   2.3714821   85.767524   -0.244421
    # 2025-Jun-11 12:57 *    52.365464  14.766261    0.0000 /?   2.3713593   85.768347   -0.242670
    #R 2025-Jun-11 12:58 *    52.208722  14.939158    0.0000 /?   2.3712368   85.769176   -0.240907
    # 2025-Jun-11 12:59 *    52.051420  15.111689    0.0000 /?   2.3711147   85.770010   -0.239133

    # refraction
    #I 2025-Jun-10 20:25 *   308.044180  15.346975  09 00 00.1716    0.0000 /?   2.3698420   86.004757   0.9846484
    #C 2025-Jun-10 20:26 *   307.886433  15.175285  09 01 00.3359    0.0000 /
    #R 2025-Jun-10 20:27 *   307.729248  15.003242  09 02 00.5001    0.0000 /?   2.3700755   86.007259   0.9881320
    # 2025-Jun-10 20:28 *   307.572622  14.830849  09 03 00.6644    0.0000 /?   2.3701928   86.008503   0.9898559
    #
    #C 2025-Jun-11 12:54 *    52.832369  14.309867  01 31 42.6456    0.0000 /?   2.3717287   85.765896   -0.247887
    # 2025-Jun-11 12:55 *    52.677285  14.483088  01 32 42.8099    0.0000 /?   2.3716052   85.766707   -0.246160
    #I 2025-Jun-11 12:56 *    52.521651  14.655966  01 33 42.9742    0.0000 /?   2.3714821   85.767524   -0.244421
    # 2025-Jun-11 12:57 *    52.365464  14.828500  01 34 43.1385    0.0000 /?   2.3713593   85.768347   -0.242670
    #R 2025-Jun-11 12:58 *    52.208722  15.000687  01 35 43.3028    0.0000 /?   2.3712368   85.769176   -0.240907
    # 2025-Jun-11 12:59 *    52.051420  15.172522  01 36 43.4671    0.0000 /?   2.3711147   85.770010   -0.239133

