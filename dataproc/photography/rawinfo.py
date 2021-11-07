import glob
import astropy.units as u
import astropy.time as apt
import matplotlib.pyplot as plt
import exifread as er
import numpy as np
import pandas as pd

fields = ['EXIF FNumber', 'EXIF ExposureTime',
          'EXIF ISOSpeedRatings', 'EXIF DateTimeOriginal']

df = pd.DataFrame()

dirname = "d:/dcim/102ND750/"
image_list = glob.glob(dirname+"/*nef")

i = 0
print(f"Procesing {len(image_list)} images: ", end='')
for image in image_list:
    tags = er.process_file(open(image, 'rb'))
    vals = [tags[f].printable for f in fields]

    data = {'fnumber': eval(vals[0]),
            'exposure': vals[1],
            'exposure_sec': eval(vals[1]),
            'iso': eval(vals[2]),
            'date': apt.Time(vals[3].replace(":", "-", 2))
            }
    df = df.append(data, ignore_index=True)
    if i % 10 == 9:
        print('x', end='')
    else:
        print(".", end="")
    i += 1
print("")

reftime = df['date'][0]

df['delta_time'] = [(dt-reftime).to(u.s).value for dt in df['date']]
df['delta_post'] = df['delta_time'] + df['exposure_sec']
df['ev'] = np.log(100*df['fnumber']**2/df['exposure_sec']/df['iso'])/np.log(2)

delta_time_2 = [item for sublist in zip(df['delta_time'], df['delta_post']) for item in sublist]
exposure_sec_2 = [item for item in df['exposure_sec'] for _ in (1, 2)]

f, ax = plt.subplots(1, 1)
ax.plot(df['delta_time'], df['ev'], '.r')
ax2 = ax.twinx()
ax2.semilogy(delta_time_2, exposure_sec_2, color='b')
ax.set_ylabel("ev")
ax.yaxis.label.set_color('red')
ax2.set_ylabel("exposure")
ax2.yaxis.label.set_color('blue')

f.tight_layout()
plt.show()
