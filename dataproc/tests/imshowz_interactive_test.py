import dataproc as dp

test = '/home/raw/tramos/dk154/20221106/K2_138_000135.fits'

af = dp.AstroFile(test)

ims = af.imshowz(interactive=True)

print(ims)
dt = af.reader()

