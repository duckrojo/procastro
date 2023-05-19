

import procastro.photography as pap
import astropy.units as u

basedir = "C:/Users/duckr/OneDrive/Documents/eclipse/rxs/"
files = ["1_0+4_0", "1_0+2_0",
         "0_7+1_6+x3", "0_8+1_6+x3+2",
         "0_9+1_6+x3+2_3", "1_0+1_6+x3+2_5",
         "r1_0+1_6+x3+2_5",
         ]
ref_time = ["2021-11-20 01:33", "2021-11-20 17:01",
            "2021-11-20 17:54", "2021-11-20 18:09:37",
            "2021-11-20 18:16:04", "2021-11-20 18:36:02",
            "2021-11-20 18:44",
            ]

item = 6

out = pap.ReadScript(f"{basedir}{files[item]}/rxs.csv",
                     ref_time=ref_time[item],
                     name=files[item])

out.plot_ev(xlims=[-15, 50], ax=1, marker='v',
            label="Orchestrator")
#out.plot_exposure(xlims=[-20, 60], ax=2)

raw = pap.RawFiles(f"{basedir}{files[item]}",
                   name=files[item],
                   hour_offset=-0.052, extension="cr2",
                   ref_time=ref_time[item]).read_exif()
raw.plot_ev(ax=1, marker='^', color='blue', legend=True,
            label="Camera")
# raw.plot_exposure(ax=2)

# raw.show()
