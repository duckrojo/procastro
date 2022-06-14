from dataproc.obsrv import AvailableAt
import matplotlib.pyplot as plt


a = AvailableAt('La Silla Observatory',
                min_obs_ix=25.0,
                night_angle=-12)
a.run_day('2022-07-01').plot()
a.run_day('2022-07-02').plot()
a.run_day('2022-07-09').plot()
a.run_day('2022-07-10').plot()
a.run_day('2022-07-11').plot()
a.run_day('2022-07-18').plot()
a.run_day('2022-07-19').plot()
a.run_day('2022-07-20').plot()


plt.show()