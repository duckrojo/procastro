import matplotlib.pyplot as plt
import procastro as pa
import astropy.coordinates as apc
import astropy.time as apt
import astropy.units as u
import numpy as np
from scipy.spatial.transform import Rotation as R

f, ax = pa.figaxes()
f.tight_layout()
ax.set_aspect('equal')

#time = apt.Time("2022-01-02T18:33:00", format='isot', scale='utc') # new Moon
time = apt.Time("2022-08-09T23:30:00", format='isot', scale='utc')  #N1
time = apt.Time("2022-08-12T01:00:00", format='isot', scale='utc')  #N2
time = apt.Time("2022-08-15T06:30:00", format='isot', scale='utc')  #N3
span_time_hours = 3.6
time2 = time+10*u.min
site = apc.EarthLocation.of_site("lco")
moon = apc.get_moon(time, location=site)
moon2 = apc.get_moon(time2, location=site)
sun = apc.get_sun(time)
moon_radius_physical = 3478.8*u.km/2

moon_distance = moon.distance
moon_radius = ((moon_radius_physical / moon_distance) * 180 * u.deg / np.pi).to(u.arcsec)


def add_label(text, to_earth, azimuth, offset_label=-200, ha="left", va="center", **kwargs):
    coordinates = ([(moon_radius * np.sin(to_earth) * np.sin(azimuth)).value,
                    (moon_radius * np.sin(to_earth) * np.cos(azimuth)).value])
    ax.plot(*coordinates, 'o', **kwargs)
    ax.annotate(text, coordinates, (coordinates[0], coordinates[1]+offset_label),
                ha=ha, va=va,
                #arrowprops={'arrowstyle': '->'},
                **kwargs)

def add_label_xy(text, xyz, offset_label, **kwargs):

    ax.annotate(f"{text}{'$_{{bhd}}$' if xyz[2]<0 else ''}", xyz[[0, 1]], (xyz[0]+offset_label[0],
                                                                           xyz[1]+offset_label[1]),
                arrowprops={'arrowstyle': '->'}, **kwargs)


def sub_earth_n_position_angle(body):
    elongation = moon.separation(body)
    to_sub_earth_angle = (np.arctan2(body.distance * np.sin(elongation),
                                     moon_distance -
                                     body.distance * np.cos(elongation)))

    print(to_sub_earth_angle.to(u.deg))
    position_angle = moon.position_angle(body)

    return to_sub_earth_angle, position_angle

def shadow(delta, azimuth, **kwargs):

    xy = []
    for angle in np.linspace(0, 360, 50)*u.deg:
        coo = np.array([0, np.cos(angle), np.sin(angle)])*u.deg
        ncoo = R.from_euler("y", -delta.to(u.deg).value, degrees=True).apply(coo)
        nncoo = R.from_euler("x", -azimuth.to(u.deg).value, degrees=True).apply(ncoo)
        nncoo *= moon_radius.value
        if nncoo[0] > 0:
            xy.append(nncoo[[1, 2]])

    ax.plot(*list(zip(*xy)), **kwargs)

def xyz_from_sidereal(skycoord):

    coo = skycoord.cartesian.get_xyz()
    ncoo = R.from_euler("z", moon.ra.to(u.deg).value, degrees=True).apply(coo)
    nncoo = R.from_euler("y", -moon.dec.to(u.deg).value, degrees=True).apply(ncoo)

    return nncoo[[1, 2, 0]]*moon_radius.value

def plot_parallactic_angle(time, span, delta, correction_deg=30):
    par = np.array([moon_radius.value*1.02, moon_radius.value*1.08])
    times = time + np.arange(0, span*60, delta.value)*u.min
    idx_color = (1*u.hour/delta).to(u.dimensionless_unscaled).value

    mn = apc.get_moon(times, site)
    altaz = mn.transform_to(apc.AltAz(obstime=times, location=site))
    ha = times.sidereal_time("mean", longitude=site.lon) - mn.ra
    par_ang = np.arctan2(np.sin(ha)*np.cos(site.lat),
                         (np.cos(mn.dec)*np.sin(site.lat) - np.cos(site.lat)*np.sin(mn.dec)*np.cos(ha)))
    #par_ang = np.arcsin(np.sin(ha)*np.cos(site.lon)/np.cos(altaz.alt))
    #par_ang = np.arcsin(np.sin(altaz.az)*np.cos(site.lon)/np.cos(mn.dec))

    print(f"lon{90*u.deg-site.lon} dec{90*u.deg-mn.dec}")
    for t,h,a,z,p in zip(times,ha,altaz.alt,altaz.az,par_ang.to(u.deg)):
        print(t,h,a,z,p)
    print(f"lon{90*u.deg-site.lon} dec{90*u.deg-mn.dec}")

    for ang, t in zip(par_ang+correction_deg*u.deg, times):
        plt.plot(par*np.sin(ang), par*np.cos(ang), color="blue")
        tm = t.ymdhms
        if tm[4] == 0:
            ax.plot(par[1] * np.sin(ang).value, par[1] * np.cos(ang).value, "or")
            ax.annotate(f"{tm[3]}", (1.04 * par[1] * np.sin(ang).value, 1.04 * par[1] * np.cos(ang).value),
                                    size=7, ha='center', va='center', rotation=-ang.to(u.deg).value)

    col = np.arange(len(times)) % idx_color == 0
    print(idx_color)
    plt.plot(par[1]*np.sin(par_ang[0]+correction_deg*u.deg),
             par[1]*np.cos(par_ang[0]+correction_deg*u.deg),
             "o", color="black", markersize=5)




angles = np.linspace(0, 360, 50)*u.deg
ax.plot(moon_radius*np.sin(angles), moon_radius*np.cos(angles))
ax.arrow(0, 0, ((moon2.ra-moon.ra)*np.cos(moon.dec)).to(u.arcsec).value, (moon2.dec-moon.dec).to(u.arcsec).value,
         length_includes_head=True, width=20)
ax.annotate("10min displacement", (0, -100))

to_sub_earth_angle, position_angle = sub_earth_n_position_angle(sun)
post = ')' if to_sub_earth_angle > 90*u.deg else ''
pre = '(' if to_sub_earth_angle > 90*u.deg else ''
add_label(f"{pre}Sub-solar{post}",
          to_sub_earth_angle, position_angle, -moon_radius.value/9, color="red", ha='center')
shadow(to_sub_earth_angle, position_angle, color="red")

ax.annotate(f"R$_L$ {moon_radius.value:.1f}''",
            (0.98*moon_radius.value, 0.9*moon_radius.value),
            ha="right")
ax.annotate(f"{100-to_sub_earth_angle.to(u.deg).value*100/180:.1f}% illum",
            (-0.98*moon_radius.value, 0.9*moon_radius.value),
            )
ax.annotate(f"$\\alpha$ {moon.ra.to(u.hourangle).value:.1f}$^h$",
            (-0.95*moon_radius.value, -0.95*moon_radius.value),
            )
ax.annotate(f"$\\delta$ {'+' if moon.dec.value > 0 else ''}"
            f"{moon.dec.to(u.deg).value:.1f}$^\circ$",
            (0.95*moon_radius.value, -0.95*moon_radius.value),
            ha="right")


add_label_xy(f"radiant", xyz_from_sidereal(apc.SkyCoord("3h +58d00")), (-moon_radius.value/5,
                                                                        -moon_radius.value/5))


ax.set_title(time)

ax.annotate("N", (0, .95*moon_radius.value), ha="center", va="center")
ax.annotate("E", (.95*moon_radius.value, 0), ha="center", va="center")
ax.annotate("S", (0, -.95*moon_radius.value), ha="center", va="center")
ax.annotate("W", (-.95*moon_radius.value, 0), ha="center", va="center")

plot_parallactic_angle(time, span_time_hours, 5*u.min)

