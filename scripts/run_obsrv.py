import procastro.obsrv as pao
import matplotlib
matplotlib.use('TkAgg')

# it is important to assign the following to a variable, otherwise plot will not be interactive due to
# garbage collection of event handler
a = pao.Obsrv("wasp-20 b", timespan=2025)
