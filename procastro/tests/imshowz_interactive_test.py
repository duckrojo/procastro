import procastro as pa

test = '/home/raw/tramos/dk154/20221106/K2_138_000135.fits'

af = pa.AstroFile(test)

ims = af.imshowz(interactive=True)

print(ims)
dt = af.reader()

