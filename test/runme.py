#!/usr/bin/python

import dataproc as dp
import pyfits as pf


testfile='test/test.fits'
data=pf.open(testfile)[0].data
exam = dp.examine2d(data)


