from dataproc.obsrv import AvailableAt
import matplotlib.pyplot as plt


a = AvailableAt('La Silla Observatory',
                min_obs_ix=25.0,
                night_angle=-12)
a.run_day('2022-07-01').plot(50)
a.run_day('2022-07-02').plot(50)
a.run_day('2022-07-09').plot(50)
a.run_day('2022-07-10').plot(50)
a.run_day('2022-07-11').plot(50)
a.run_day('2022-07-18').plot(50)
a.run_day('2022-07-19').plot(50)
a.run_day('2022-07-20').plot(50)


plt.show()