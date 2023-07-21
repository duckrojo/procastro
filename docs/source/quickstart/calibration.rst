Calibration Files
=================

.. code-block:: python

  import procastro as pa

  master_bias = pa.AstroDir("*.fits").filterfilter(object="bias").median()

  master_flat = pa.AstroDir("*.fits", mbias=master_bias).filter(object="flat").median(normalize=True)

  science_frames = pa.AstroDir("*.fits", mbias=master_bias, mflat=master_flat).filter(exptime_gt=40, object_ne="flat")

if mbias or mflat keywords are used, then all the operations on data will have those calibrations applied automatically.

