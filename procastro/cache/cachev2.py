import copy
import os
import pickle
import tempfile
from typing import Optional
import queue
import diskcache as dc

__all__ = ['astrofile_cache', 'jpl_cache', 'usgs_map_cache']


from procastro.misc.misc_general import user_confdir
import astropy.time as apt
import astropy.units as u


class _AstroCachev2:
    def __init__(self,
                 max_cache=200, lifetime=0,
                 hashable_kw=None, label_on_disk=None,
                 force: str | None = None,
                 ):
        """
        New implementation of the cache system, using diskcache
        Parameters
        ----------
        force :
          Keyword which, if True, will force to recompute the cache.
        max_cache
          how many days to cache in disk, if more than that has elapsed, it will be reread.
        """

        self._max_cache: int = max_cache
        self.lifetime = lifetime
        self.force = force

        if label_on_disk is not None:
            if any((not_permitted in label_on_disk) for not_permitted in ('/', ':')):
                raise ValueError(f"label_on_disk contains invalid file characters '{label_on_disk}'")
            self._store_on_disk = True
        else:
            self._store_on_disk = False

        if self._store_on_disk:
            # NOTE: Diskcache handles the cache directory and config file automatically, so there is 
            # no need to create them manually.
            self.cache_directory = user_confdir(f'cache/{label_on_disk}', use_directory=True)
            self._cache = dc.Cache(self.cache_directory, size_limit=max_cache) 

        else:
            # NOTE: Diskcache handles the cache directory and config file automatically, so there is
            # no need to create them manually.
            self._cache = dc.Cache(size_limit=max_cache)
        ## NOTE: The sorted_hash and queue logic is not needed with diskcache, as it handles the cache automatically.
        ## The cache is automatically sorted by the time of access, so there is no need to manually sort it.
        if hashable_kw is None:
            hashable_kw = []
        self._hashable_kw = hashable_kw

    def __bool__(self):
        return self._max_cache > 0

    def available(self):
        return not self._queue.full()

    def _delete_cache(self):
        # NOTE: Diskcache handles the cache automatically, so there is no need to manually delete it.
        if len(self._cache)> 0:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    def _store_cache(self, compound_hash, content):
        # NOTE: Diskcache handles the cache automatically, so there is no need to manually store it.
        self._cache.set(
            compound_hash,
            content,
            expire=self.lifetime * 86400 if self.lifetime else None  # Convert days to seconds
        )
        
    def __call__(self, method):
        def wrapper(hashable_first_argument, **kwargs):
            """Instance is just the first argument to the function which would need a __hash__:
             in a method it refers to self."""
            cache = True
            compound_hash = tuple([hashable_first_argument] +
                                  [kwargs[kw] for kw in self._hashable_kw])

            if kwargs.pop(self.force, False):
                cache = False

            try:
                if compound_hash in self._cache:
                    pass
            except TypeError:
                # disable cache if type is not hashable (numpy array for instance)
                cache = False

            if cache and (compound_hash in self._cache):
                if self.lifetime:
                    cached_time = apt.Time(self._cache.get(compound_hash + ('_time',)), format='isot', scale='utc')
                    if apt.Time.now() - cached_time > self.lifetime * u.day:
                        self._delete_cache()
                    else:
                        return self._cache[compound_hash]
                else:
                    return self._cache[compound_hash]

            ret = method(hashable_first_argument, **kwargs)

            # save if caching
            if cache and self._queue is not None:

                # delete oldest cache if limit reached
                if self._queue.full():
                    self._delete_cache()

                self._store_cache(compound_hash, ret)

            return ret

        return wrapper

    def set_max_cache(self, max_cache: int):
        if max_cache < 0:
            raise ValueError(f"max_cache must be positive ({max_cache})")

        self._max_cache = max_cache
        self._cache.cull()  # Reduce cache size if necessary


astrofile_cache = _AstroCachev2()
jpl_cache = _AstroCachev2(max_cache=50)
usgs_map_cache = _AstroCachev2(max_cache=30, lifetime=30,
                             hashable_kw=['detail'], label_on_disk='USGSmap',
                             force="no_cache")

