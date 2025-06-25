import os

import numpy as np
from matplotlib import pyplot as plt

import procastro
from procastro.astro import find_time_for_altitude, moon_distance, query_transit_ephemeris, get_transit_ephemeris_file

from astropy import table
import astropy.coordinates as apc
import astropy.units as u
import astropy.time as apt


class ExoPlanet:
    def __init__(self, name, timespan,
                 site="lasilla", twilight=-12, equinox="J2000",
                 min_altitude: float = 30,
                 min_hours: float = 2,
                 min_moon: float = 10,
                 ):
        self.params = {'equinox': equinox,
                       'twilight': twilight,
                       'name': name,
                       'site': site,
                       'min_altitude': min_altitude,
                       'min_hours': min_hours,
                       'min_moon': min_moon,
                       }
        self.transit_info = {}
        self.target = self.set_target(name)
        self.location = self.set_site(site)
        self.timespan = self.set_timespan(timespan)
        self.phase_info = self.compute_daily_phases()

    def sort_by_coverage(self, min_phase, max_phase,
                         min_coverage=0.5):

        phase_from = (self.phase_info['min_phases'].data - min_phase) % 1
        phase_to = (self.phase_info['max_phases'].data - min_phase) % 1
        max_phase = np.ones(len(phase_to))*((max_phase - min_phase) % 1)
        period = self.transit_info['period']
        vis_from = self.phase_info['from']
        vis_to = self.phase_info['to']

        moon = self.phase_info['moons'].data

        begin_coverage = np.min([max_phase, phase_from], 0)
        begin_coverage[phase_to < phase_from] = 0

        end_coverage = np.min([max_phase, phase_to], 0)

        coverage = (end_coverage - begin_coverage) / max_phase
        coverage = coverage*(vis_from < vis_to)

        sorted_idx = np.argsort(coverage)

        phase_range = ((max_phase[0])*period).to(u.min).value
        print(f"\nBest coverage for phase range: {min_phase} - {max_phase[0]+min_phase} "
              f"({phase_range//60:.0f}h{phase_range%60:.1f}m)")
        delta_from = 1 - phase_from
        delta_from[phase_to > phase_from] = 0
        delta_to = phase_to - max_phase
        delta_to = delta_to * (delta_to > 0)
        for idx in sorted_idx[::-1]:
            if coverage[idx] < min_coverage:
                break
            if moon[idx] < self.params['min_moon']:
                continue

            vis_from = self.phase_info['from'][idx]
            vis_to = self.phase_info['to'][idx]

            print(f"{coverage[idx]*100:3.0f}%: {(vis_to - vis_from).to(u.hour):.1f}: "
                  f"{'<' if delta_from[idx] else '|'}{(vis_from + delta_from[idx]*period).isot[:-7]} - "
                  f"{(vis_to - delta_to[idx]*period + 0.99*u.min).isot[:-7]}{'>' if delta_to[idx] else '|'}"
                  f" {f' *moon{moon[idx]:.0f}' if moon[idx] < 30 else ''}")

    def compute_daily_phases(self):
        t0, t1 = self.timespan
        ndays = int((t1-t0).to(u.day).value)

        twilight = self.params['twilight']
        min_altitude = self.params['min_altitude']

        midday = t0 + self.params["to_local_midday"] * u.day
        midnight0 = find_time_for_altitude(self.location, midday,
                                           ref_altitude_deg="min", find="next", body="sun")
        midnight_sidereal0 = midnight0.sidereal_time('apparent', self.location).hourangle

        delta_highest = ((self.target.ra.hourangle - midnight_sidereal0) / 24 % 1)
        highest = midnight0 + (delta_highest - (delta_highest > 0.5)) * u.sday
        hdelta_star = find_time_for_altitude(self.location, highest, min_altitude,
                                             find="next", body=self.target) - highest

        time0 = self.transit_info['epoch']
        period = self.transit_info['period']

        delta_sidereal = (1*u.day - 1*u.sday)

        days = np.arange(ndays)
        midnights = midnight0 + days * u.day
        midnights_sidereal = midnight_sidereal0 + days * delta_sidereal.to(u.hour).value % 24

        print("- computing sunsets")
        sunsets = [find_time_for_altitude(self.location, midnight,
                                          ref_altitude_deg=twilight, find="prev", body="sun")
                   for midnight in midnights]
        print("- computing sunrises")
        sunrises = [find_time_for_altitude(self.location, midnight,
                                           ref_altitude_deg=twilight, find="next", body="sun")
                    for midnight in midnights]

        print("- computing moon and others")
        delta_highest = ((self.target.ra.hourangle - midnights_sidereal) / 24 % 1)
        highests = midnights + (delta_highest - (delta_highest > 0.5)) * u.sday

        starrises = highests - hdelta_star
        starsets = highests + hdelta_star

        vis_from = apt.Time(np.max([starrises, sunsets], axis=0))
        vis_to = apt.Time(np.min([starsets, sunrises], axis=0))

        moons = np.array([moon_distance(self.target, location=self.location, obs_time=midnight).value
                          for midnight in midnights])

        min_phases = (vis_from - time0) / period % 1
        central_phases = (highests - time0) / period % 1
        max_phases = (vis_to - time0) / period % 1

        proximity = np.array(["green"] * ndays, dtype="<U10")
        proximity[moons < 30] = "yellow"
        proximity[moons < 15] = "red"

        print("= done computing")
        return table.Table({'from': vis_from,
                            'to': vis_to,
                            'moons': moons,
                            'min_phases': min_phases,
                            'central_phases': central_phases,
                            'max_phases': max_phases,
                            'highests': highests,
                            'proximity': proximity,
                            })

    def plot_phases(self, ax=None,
                    shade=None,
                    show=False,
                    ):
        t0, t1 = self.timespan
        ndays = int((t1-t0).to(u.day).value)

        twilight = self.params['twilight']
        min_altitude = self.params['min_altitude']
        min_hours = self.params['min_hours']*u.hour
        min_moon = self.params['min_moon']

        visibles = self.phase_info['to'] - self.phase_info['from']
        moons = self.phase_info['moons']

        filtered = visibles > min_hours
        filtered *= moons > min_moon

        month = ["Jan", "Feb", "Mar", "Apr", "May",
                 "Jun", "Jul", "Aug", "Sep",
                 "Oct", "Nov", "Dec"]
        if ax is None:
            ax = plt.subplot(111)
            ax.cla()

        bom = []
        for day, (min_phase, central_phase, max_phase, highest, color)\
            in enumerate(self.phase_info[['min_phases', 'central_phases', 'max_phases', 'highests', 'proximity']]):
            y,mn,d,h,m,s = highest.ymdhms
            if not filtered[day]:
                continue

            if min_phase > max_phase:
                ax.fill_between([day, day + 1], 0, max_phase, color=color, alpha=0.5, zorder=2)
                ax.fill_between([day, day + 1], min_phase, 1, color=color, alpha=0.5, zorder=2)
            else:
                ax.fill_between([day, day + 1], min_phase, max_phase, color=color, alpha=0.5, zorder=2)
            ax.text(day + 0.5, central_phase, f"{mn:02}-{d:02}T{h:02}:{m:02}",
                    ha='center', va='center', color='black', rotation=90, zorder=3)

            y,mn,d,h,m,s = (t0+day*u.day).ymdhms
            if d == 1:
                bom.append([day, mn, y])
                ax.axvline(day, color="purple", ls="dashed", alpha=0.5, zorder=3)

        vals = [b[0] for b in bom]
        names = [month[b[1]-1]+str(b[2]) for b in bom]
        ax.xaxis.set_ticks(vals, names)
        ax.set_xlim(0, ndays)

        if shade is not None:
            if shade[0] < shade[1]:
                ax.axhspan(shade[0], shade[1], color="gray", alpha=0.5, zorder=0)
            else:
                ax.axhspan(shade[0], 1, color="gray", alpha=0.5, zorder=0)
                ax.axhspan(0, shade[1], color="gray", alpha=0.5, zorder=0)
            phase_range = ((shade[1]-shade[0]) * self.transit_info['period']).to(u.min).value
            msg = f"Shading {shade[0]} - {shade[1]} ({phase_range // 60:.0f}h{phase_range % 60:.1f}m)"
            self.sort_by_coverage(shade[0], shade[1])
        else:
            msg = ""

        ax.set_ylabel(f"Orbital phase for {self.params['name']}. {msg}")
        ax.set_xlabel(f"Observability from {self.params['site']}")
        ax.set_title(f"Min altitude: {min_altitude:.0f}$^\\circ$, max twilight: {twilight:.0f}$^\\circ$. "
                     f"Showing times of maximum altitude")

        if show:
            ax.figure.show()

    def set_timespan(self, timespan, samples=120,
                     central_time=25,
                     **kwargs):
        """
        Set time span

        Parameters
        ----------
        timespan : str, int
        samples : int, optional
        central_time : int
        """
        if timespan is None:
            if 'timespan' in self.params:
                timespan = self.params["timespan"]
            else:
                raise ValueError("Timespan needs to be specified")
        else:
            self.params["timespan"] = timespan

        if isinstance(timespan, int):   # Year
            # times always at midnight (UT)
            t0 = apt.Time('{0:d}-01-01'.format(timespan,))
            t1 = apt.Time('{0:d}-01-01'.format(timespan+1,))

        elif isinstance(timespan, str):  # Start Year / End Year
            years = timespan.split('/')
            if len(years) != 2:
                raise NotImplementedError(
                    "Requested timespan ({0:s}) is not valid. Only a string "
                    "in the format <FROM/TO> is accepted (only one "
                    "slash separating dates)".format(years[0]))
            try:
                t0 = apt.Time('{0:d}-1-1'.format(int(years[0]), ))
            except ValueError:
                t0 = apt.Time(years[0])
            try:
                t1 = apt.Time('{0:d}-1-1'.format(int(years[1]) + 1, ))
            except ValueError:
                t1 = apt.Time(years[1])

        else:
            raise NotImplementedError(
                "Requested timespan ({0:s}) not implemented "
                "yet. Currently supported: * single "
                "integer (year)".format(timespan,))

        print(f"working timespan: {t0} - {t1}")

        return t0, t1

    def set_site(self, site):
        """
        Define name from list available from EarthLocation

        Parameters
        ----------
        site: string ,optional
            Identifies the observatory as a name, or (lat, lon) tuple,
        """

        location = apc.EarthLocation.of_site(site)

        # Parameters for user-friendly values
        self.params["site"] = site
        # fraction of day to move from UT to local midday
        self.params["to_local_midday"] = 0.5 - location.lon/(360.0*u.deg)

        print(f"Selected (lat, lon): {location.lat}, {location.lon}")

        return location

    def set_target(self, target, magnitude=10, star_name='',
                   transit_epoch=None, transit_period=None, transit_length=1,
                   phase_offset=0.0,
                   **kwargs):
        """
        Set star and site into pyephem

        Parameters
        ----------
        target:
            Either RA and Dec in hours and degrees, or target name
            to be queried
        magnitude:
        star_name: string, optional
            Name of host star
        transit_length: float, optional
        transit_period: float, optional
        transit_epoch: float, optional
        phase_offset: float, optional
            Set to 0.5 to show occultations instead of transits
        """
        self.params['given_target'] = target
        # if 'current_transit' in self.params:
        #     del self.params['current_transit']

        paths = [os.path.dirname(__file__)+'/coo.txt',
                 os.path.expanduser("~")+'/.coostars'
                 ]
        target_obj = procastro.astro.coordinates.find_target(target,
                                                             coo_files=paths,
                                                             equinox=self.params["equinox"])

        print("Star at RA/DEC: {0:s}/{1:s}"
              .format(target_obj.ra.to_string(sep=':'),
                      target_obj.dec.to_string(sep=':')))

        ephemeris = get_transit_ephemeris_file(target)

        if not all(ephemeris):
            ephemeris = query_transit_ephemeris(target)
            print("Found in Database: ", end="")
        else:
            print("Found in File: ", end="")

        transit_epoch, transit_period, transit_length = ephemeris
        print(f"{transit_epoch} + E*{transit_period} +- {transit_length}")

        self.transit_info = {'length': transit_length * u.hour,
                             'epoch': apt.Time(transit_epoch, format='jd'),
                             'period': transit_period * u.day,
                             }
        return target_obj
