import matplotlib
import procastro.obsrv as pao

matplotlib.use('TkAgg')

print("Initializing...")
night = pao.Nightly()

print(" plotting...")
night.plot("2025-05-06")

night.show()
# night.savefig("nightly.png")

