from dataproc.obsrv import AvailableAt
import matplotlib.pyplot as plt


a = AvailableAt('2021-10-09', 'La Silla Observatory',
                min_obs_ix=25.0,
                night_angle=-12)
a.plot(50, extention=False)

plt.show()