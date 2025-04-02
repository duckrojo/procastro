import os
import diskcache as dc
from typing import Optional
import astropy.time as apt
import astropy.units as u
from procastro.misc.misc_general import user_confdir

__all__ = ['astrofile_cache', 'jpl_cache', 'usgs_map_cache']


class _AstroCachev2:
    """
    _AstroCachev2 is a caching utility class designed to manage and store large astronomical data efficiently 
    with support for in-memory and disk-based caching. It provides mechanisms for cache 
    eviction, expiration, and retrieval based on hashable keys.
    Attributes:
        max_cache (int): The maximum volume allocated for the cache in Bytes.
        expire (int): The expire time of cached items in days. If 0, items do not expire.
        force (str | None): A keyword to force bypassing the cache when set in method arguments (forces execution).
        eviction_policy (str | None): The policy used to evict items from the cache.

            "least-recently-stored" is the default. Every cache item records the time it was stored in the cache. This policy adds an index to that field. On access, no update is required. Keys are evicted starting with the oldest stored keys. As DiskCache was intended for large caches (gigabytes) this policy usually works well enough in practice.

            "least-recently-used" is the most commonly used policy. An index is added to the access time field stored in the cache database. On every access, the field is updated. This makes every access into a read and write which slows accesses.

            "least-frequently-used" works well in some cases. An index is added to the access count field stored in the cache database. On every access, the field is incremented. Every access therefore requires writing the database which slows accesses.

            "none" disables cache evictions. Caches will grow without bound. Cache items will still be lazily removed if they expire. The persistent data types, Deque and Index, use the "none" eviction policy. For lazy culling use the cull_limit setting instead.
        store_on_disk (bool): Indicates whether the cache is stored on disk.
        cache_directory (str): The directory path for disk-based caching (if enabled).
        hashable_kw (list): A list of keyword arguments that are hashable and used 
            to generate compound hash keys.
    
    """
    def __init__(self,
                 max_cache=int(1e6), lifetime=0,
                 hashable_kw=None, label_on_disk=None,
                 force: str | None = "force",
                 eviction_policy: str | None = 'least-recently-used'):
        
        self.max_cache: int = max_cache
        self.lifetime = lifetime
        self.force = force
        self.eviction_policy = eviction_policy

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
            self.__cache = dc.Cache(
                self.cache_directory, size_limit=self.max_cache, eviction_policy=eviction_policy)
        else:
            self.__cache = dc.Cache(size_limit=self.max_cache, eviction_policy=eviction_policy)

        if hashable_kw is None:
            hashable_kw = []
        self.hashable_kw = hashable_kw


    def __call__(self, method):
        #METHOD to use when force= False, when the cache is not bypassed.
        @self.__cache.memoize(expire=self.lifetime*86400 if self.lifetime else None)
        def cached_method(*args,**kwargs):
            return method(*args,**kwargs)
        
        def wrapper(hashable_first_argument, **kwargs):
            """
            Wrapper to handle custom logic like bypassing the cache.
            """
            # Check if caching is disabled via the `force` keyword
            if kwargs.pop(self.force, False):
                return method(hashable_first_argument, **kwargs)

            # Handle non-hashable arguments (disable caching)
            try:
                compound_hash = tuple([hashable_first_argument] +
                                      [kwargs[kw] for kw in self.hashable_kw])
                hash(compound_hash)  # Ensure the key is hashable
            except TypeError:
                raise TypeError(
                    f"Non-hashable argument detected: {hashable_first_argument}")

            # Call the memoized method, if the force parameter is false
            return cached_method(hashable_first_argument, **kwargs)

        return wrapper



astrofile_cache = _AstroCachev2()
jpl_cache = _AstroCachev2(max_cache=50)
usgs_map_cache = _AstroCachev2(max_cache=30, lifetime=30,
                               hashable_kw=['detail'], label_on_disk='USGSmap',
                               force="no_cache")
