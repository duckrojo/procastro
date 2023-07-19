
Working with Multiple Files
=======

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
