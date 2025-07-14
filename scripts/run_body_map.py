import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

import astropy.time as apt
import astropy.units as u

from procastro.astro import body_map

##########################################
#
extra_radius = 30*u.arcmin
max_separation_center = 10*u.arcmin
starting_time = apt.Time.now()
delta_time = 229.5*u.day
#
#######################################


times = starting_time + np.linspace(0, delta_time, 80)
# times = (starting_time + 216*u.day)
# times = starting_time
# times = starting_time + 229*u.day


# path = path_body("moon",
#                  "lasilla",
#                  times,
#                  use_jpl=True,
#                  )
#
# path_cover = polygon_around_path(path['skycoord'],
#                                  max_separation_center + extra_radius,
#                                  close=True,
#                                  )
# path_with_moon = polygon_around_path(path['skycoord'],
#                                      15 * u.arcmin,
#                                      close=True,
#                                      )
#
# stars_outside = simbad_between_polygons(path_cover, path_with_moon,
#                                         brightest=5,
#                                         dimmest=11,
#                                         filter_name='V',
#                                         )
# stars_inside = simbad_between_polygons(path_with_moon,
#                                        brightest=5,
#                                        dimmest=11,
#                                        filter_name='V',
#                                        )
#
#
# ax = starry_plot([stars_inside, stars_outside],
#                  areas=[path_with_moon, path_cover],
#                  stars_color=["blue", "red"],
#                  )

ret = body_map("mars", "lasilla", times, detail="Color Viking",
#               color_background='white',
#               radius_to_plot=1000,
#               reread_usgs=True,
                locations={"tycho": (348.68, -43.37),  # tycho crater
                     },
#               filename="moon.mpg",
                fps=30, dpi=100,
                verbose=True
                )