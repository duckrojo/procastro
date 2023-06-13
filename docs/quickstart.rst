Quickstart tutorial
===================


First, load the module

.. code-block:: python

  import procastro as pa


You can load files from a directory with

.. code-block:: python

  ad =  pa.AstroDir("*.fits")

and find files with easy filtering

.. code-block:: python

  # either with logical OR
  ad_or_filtered = ad.filter(name_match="wasp", exptime_lt=120)

  # or with logical AND
  ad_and_filtered = ad.filter(name_match="wasp").filter(exptime_lt=120)

also with easy indexing and mixing

.. code-block:: python

  ad_selected = ad[ad["exptime"] > 10]
  ad_selected = ad[:5] + ad[7:]
