from procastro.astro import body_map
from astropy import time as apt

ax = body_map("moon", "lco", apt.Time.now())
ax.figure.show()


