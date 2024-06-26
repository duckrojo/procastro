import warnings
from typing import Sequence

import numpy as np
from astropy import coordinates as apc, time as apt, units as u
from astropy.table import Table
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt, ticker
import cartopy.crs as ccrs
from matplotlib.patches import Polygon

import procastro as pa
from procastro.astro import aqs as aqs


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
                           ref_altitude_deg: str | float,
                           search_delta_hour: float = 2,
                           search_span_hour: float = 16,
                           fine_span_min: float = 20,
                           find: str = "next",
                           body: str = "sun",
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
       find can be: 'next', 'previous'/'prev', or 'around'
    time: apt.Time
       starting time for the search. It must be within 4 hours of the middle of day to work with default parameters.
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
    closest_idx = pa.parabolic_x(altitude_rough - ref_altitude, central_idx=central_idx, vertex=vertex)
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
                                 vertex=vertex)

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
