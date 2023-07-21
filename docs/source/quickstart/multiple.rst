
Working with Multiple Files
===========================

.. code-block:: python

  import procastro as pa


Files can be loaded from a directory with standard glob.glob wildcards.

.. code-block:: python

  ad =  pa.AstroDir("*.fits")


Find files with easy filtering function .filter() that supports handling the value of any header of the FITS file, including negate NOT, numerical comparison LT and GT, or partial string MATCH.

.. code-block:: python

  # either with logical OR
  ad_after_or_filter = ad.filter(name_match="wasp", exptime=120)

  # or with logical AND
  ad_after_and_filter = ad.filter(name_not_match="wasp").filter(exptime_lt=120)

also with easy indexing and mixing

.. code-block:: python

  ad_selected = ad[ad["exptime"] > 10]
  ad_selected = ad[:5] + ad[7:]

requesting header or data values for the whole group of files can be done with the values() or data() methods:

.. code-block:: python

  list_of_headers_per_file = ad.values(["EXPTIME", "filename", "JD-OBS"])
  array_of_nfiles_dimension = ad.get_datacube()

An AstroDir is just a collection of AstroFile that can do other tricks,

.. code-block:: python

  af = ad[0]

Depending on the indexing type, it has different returns:

.. code-block:: python

  # if using an string returns the header value
  af['exptime']

  # otherwise, it refers to the data itself
  af[100:200, 300: 450]  # if data is 2D

It also supports mathematical operations on the data directly, and logical operations on
the header previously defined as sortkey (filename by default)

.. code-block:: python

  af /= 100
  af_sum = ad[0]+ad[1]

  ad.add_sortkey("EXPTIME")
  test = ad[0] < ad[2]

Note that all the headers are read and cached upon creation of an AstroDir. Further, there
is also a cache of data up to a maximum number of files. The default of 200 cached files can be changed with

.. code-block:: python

  from procastro import astrofile_cache
  astrofile_cache.set_max_cache(100)

Whenever the data cache is full it will read from disk