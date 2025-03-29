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
                raise ValueError(f"label_on_disk contains invalid file characters '{label_on_disk}'")
            self._store_on_disk = True
        else:
            self._store_on_disk = False

        if self._store_on_disk:
            self.cache_directory = user_confdir(f'cache/{label_on_disk}', use_directory=True)
            self.__cache = dc.Cache(self.cache_directory, cull_limit=max_cache)
        else:
            
            self.__cache = dc.Cache(cull_limit=max_cache)

        if hashable_kw is None:
            hashable_kw = []
        self._hashable_kw = hashable_kw

        print(f"Created cache with size limit {self.__cache.cull_limit}")

    @property
    def _cache(self):
        """Devuelve el contenido del caché como un diccionario."""
        return {key: self.__cache.get(key) for key in list(self.__cache.iterkeys())}

    @_cache.setter
    def _cache(self, value):
        """Permite configurar el caché directamente si es necesario."""
        self.__cache = value

    def _internal_cache(self):
        """Returns the internal cache object."""
        return self.__cache
    

    def __bool__(self):
        return self._max_cache > 0

    def _delete_cache(self):
        """Deletes the oldest item in the cache."""
        while len(self.__cache) > self._max_cache:
            # Identificar la clave más antigua
            oldest_key = min(self.__cache.iterkeys(), key=lambda k: self.__cache.get(k)[0])
            # Eliminar la clave más antigua
            print(f"Deleting oldest cache entry: {oldest_key} -> {self.__cache.get(oldest_key)}")
            del self.__cache[oldest_key]

    def _store_cache(self, compound_hash, content):
        """Stores an item in the cache with an optional expiration time."""
        print(f"Storing in cache {compound_hash} -> {content}")
        self.__cache.set(
            compound_hash,
            (apt.Time.now().isot, content),  # Almacenar el tiempo y el valor como una tupla
            expire=self.lifetime * 86400 if self.lifetime!=0 else None  # Convertir días a segundos
        )
        if len(self.__cache) > self.__cache.cull_limit:
            print(f"Cache size exceeded limit: {self.__cache.cull_limit}. Deleting oldest item.")
            self._delete_cache()
        print(f"Cache size: {len(self.__cache)}")
        print(f"Stored value for {compound_hash}: {self.__cache.get(compound_hash)}")

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
                    cached_time = apt.Time(self._cache.get(compound_hash + ('_time',)), format='isot', scale='utc')
                    if apt.Time.now() - cached_time > self.lifetime * u.day:
                        self._delete_cache()
                    else:
                        return self._cache[compound_hash]
                else:
                    return self._cache[compound_hash]

            ret = method(hashable_first_argument, **kwargs)

            # Save if caching
            if cache:
                # Cull the cache if it exceeds the max size
                self.__cache.cull()
                self._store_cache(compound_hash, ret)
                # self.__cache.set(compound_hash + ('_time',), apt.Time.now().isot)

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