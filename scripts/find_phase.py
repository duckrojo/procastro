import matplotlib

from procastro.obsrv import exoplanet

matplotlib.use('TkAgg')

exoplanet.ExoPlanet("wasp76 b", "2024-09-01/2024-11-01", site="ctio",
                    ).plot_phases(shade=[0.25, 0.375])
