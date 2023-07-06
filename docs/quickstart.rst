Quickstart tutorial
===================



.. code-block:: python

  import procastro as pa


Files can be loaded from a directory with standard glob.glob wildcards.

.. code-block:: python

  ad =  pa.AstroDir("*.fits")


Find files with easy filtering function .filter() that supports handling the value of any header of the FITS file, including negate NOT, numerical comparison LT and GT, or partial string MATCH.

.. code-block:: python

  # either with logical OR
  ad_or_filtered = ad.filter(name_match="wasp", exptime=120)

  # or with logical AND
  ad_and_filtered = ad.filter(name_not_match="wasp").filter(exptime_lt=120)

also with easy indexing and mixing

.. code-block:: python

  ad_selected = ad[ad["exptime"] > 10]
  ad_selected = ad[:5] + ad[7:]

There is also a very versatile tool called imshowz that can be used interactively.
It will remind you of ds9/imexam but is Python native!

.. image:: source/figs/imshowz_z.png
  :width: 700
  :alt: zoom in area

.. image:: source/figs/imshowz_r.png
  :width: 700
  :alt: Radial profile
