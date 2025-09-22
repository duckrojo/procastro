import matplotlib

from procastro.obsrv import exoplanet

matplotlib.use('TkAgg')

exo = exoplanet.ExoPlanet("wasp-77A b", "2026-05-01/2027-04-30", site="ctio",
                    ).plot_phases(shade=[0.25, 0.375])
