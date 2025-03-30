import os
import diskcache as dc
from typing import Optional
import astropy.time as apt
import astropy.units as u
from procastro.misc.misc_general import user_confdir

__all__ = ['astrofile_cache', 'jpl_cache', 'usgs_map_cache']


class _AstroCachev2:
    def __init__(self,
                 max_cache=200, lifetime=0,
                 hashable_kw=None, label_on_disk=None,
                 force: str | None = "force"):
        """
        New implementation of the cache system, using diskcache
        """
        self._max_cache: int = max_cache
        self.lifetime = lifetime
        self.force = force

        if label_on_disk is not None:
            if any((not_permitted in label_on_disk) for not_permitted in ('/', ':')):
                raise ValueError(
                    f"label_on_disk contains invalid file characters '{label_on_disk}'")
            self._store_on_disk = True
        else:
            self._store_on_disk = False

        if self._store_on_disk:
            self.cache_directory = user_confdir(
                f'cache/{label_on_disk}', use_directory=True)
            self.__cache = dc.Cache(self.cache_directory, cull_limit=max_cache)
        else:

            self.__cache = dc.Cache(cull_limit=max_cache)

        if hashable_kw is None:
            hashable_kw = []
        self._hashable_kw = hashable_kw


    @property
    def _cache(self)-> dict:
        """
        Retrieves a dictionary representation of the current cache.
        This method constructs and returns a dictionary where the keys are the 
        same as those in the internal cache (`self.__cache`), and the values are 
        fetched from the internal cache.
        Returns:
            dict: A dictionary containing key-value pairs from the internal cache.
        """
        
        return {key: self.__cache.get(key) for key in list(self.__cache.iterkeys())}

    @_cache.setter
    def _cache(self, value):
        """
        Internal method to set the cache value.
        Args:
            value: The value to be stored in the cache.
        """

        self.__cache = value

    def __internal_cache(self) -> object: 
        """
        Accessor method for the internal cache object. This method should be only accesed by the class itself.
        It provides a way to retrieve the internal cache object without exposing it directly.
        This is useful for encapsulation and maintaining the integrity of the cache.
        It is not intended to be used outside of the class.

        Returns:
            object: The internal cache object.
        """
        return self.__cache

    def __bool__(self):
        return self._max_cache > 0

    def _delete_cache(self):
        """
        Deletes the oldest item in the cache if the cache size exceeds the maximum limit.
        We cant use the cull method provided by diskcache because it is intended when the cache is volume-full and not when it is size-full.
        This method identifies the oldest key in the cache and removes it, ensuring that the cache size remains within the defined limit.
        """
        while len(self.__cache) > self._max_cache:

            oldest_key = min(self.__cache.iterkeys(),
                             key=lambda k: self.__cache.get(k)[0])
            del self.__cache[oldest_key]

    def _store_cache(self, compound_hash, content):
        """
        Stores a given content in the cache with an associated compound hash key.
        Args:
            compound_hash (str): The unique key used to identify the cached content.
            content (Any): The data to be stored in the cache.
        Behavior:
            - The content is stored in the cache along with the current timestamp in ISO format.
            - The cache entry will expire after `self.lifetime` days if `self.lifetime` is not 0.
            - If the cache size exceeds the defined cull limit (`self.__cache.cull_limit`), 
              the oldest item in the cache is deleted to maintain the size limit.
        Note:
            - The expiration time is calculated in seconds (`self.lifetime * 86400`).
            - If `self.lifetime` is 0, the cache entry will not expire.
        """
        
        self.__cache.set(
            compound_hash,
            (apt.Time.now().isot, content),
            expire=self.lifetime * 86400 if self.lifetime != 0 else None
        )
        if len(self.__cache) > self.__cache.cull_limit:
            self._delete_cache()

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
                # Disable cache if type is not hashable (e.g., numpy array)
                cache = False

            if cache and (compound_hash in self._cache):
                
                if self.lifetime:
                    cached_entry = self._cache[compound_hash]
                    if cached_entry and isinstance(cached_entry[0], str):
                        cached_time = apt.Time(cached_entry[0], format='isot', scale='utc')
                    else:
                        raise ValueError(f"Invalid cached time format: {cached_entry}")
                    
                    if apt.Time.now() - cached_time > self.lifetime * u.day:
                        self._delete_cache()
                    else:
                        return self._cache[compound_hash]
                else:
                    return self._cache[compound_hash]

            ret = method(hashable_first_argument, **kwargs)

            # Save if caching
            if cache:
                self._store_cache(compound_hash, ret)

            return ret

        return wrapper

    def set_max_cache(self, max_cache: int):
        """Sets the maximum number of items in the cache."""
        if max_cache < 0:
            raise ValueError(f"max_cache must be positive ({max_cache})")

        self._max_cache = max_cache
        self._cache.cull()  # Reduce cache size if necessary


astrofile_cache = _AstroCachev2()
jpl_cache = _AstroCachev2(max_cache=50)
usgs_map_cache = _AstroCachev2(max_cache=30, lifetime=30,
                               hashable_kw=['detail'], label_on_disk='USGSmap',
                               force="no_cache")
