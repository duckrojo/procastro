import logging
import os
from pathlib import Path
import diskcache as dc
from procastro import config

# Imports para cache en memoria
from cachetools import TTLCache, LRUCache, LFUCache
from cachetools.keys import hashkey
import threading

logger = logging.getLogger(__name__)


class AstroCache:
    """
    AstroCache on disk (diskcache) or in memory (cachetools). 
    This class provides a caching mechanism for astronomical data, allowing for efficient retrieval and storage of results.
    Args:
        max_cache (int): Maximum size of the cache in bytes. Default is 1e9 (1 GB).
        lifetime (int): Lifetime of cached items in days. Default is 0 (no expiration).
        hashable_kw (list): List of keyword arguments that should be hashable for caching. Default is None.
        label_on_disk (str): Label for the cache directory on disk. If None, uses in-memory cache.
        force (bool): If True, bypasses the cache and forces the function to be called.
        eviction_policy (str): Eviction policy for the cache. Options are 'least-recently-used', 'least-frequently-used', or 'least-recently-stored'. Default is 'least-recently-used'.
        verbose (bool): If True, enables verbose logging for cache operations. Default is False.
    """

    def __init__(self,
                 max_cache=int(1e9), lifetime=0,
                 hashable_kw=None, label_on_disk=None,
                 force: bool = False,
                 eviction_policy: str | None = 'least-recently-used',
                 verbose=False):

        self.max_cache: int = max_cache
        self.lifetime = lifetime if lifetime is None else 86400*lifetime  # units of day
        self.force = force
        self.eviction_policy = eviction_policy
        self.verbose = verbose

        if label_on_disk is not None:
            if any((not_permitted in label_on_disk) for not_permitted in ('/', ':')):
                raise ValueError(
                    f"label_on_disk contains invalid file characters '{label_on_disk}'")
            self._store_on_disk = True

            # Diskcache
            config_dict = config.config_user(label_on_disk)
            self.cache_directory = config_dict.get('cache_dir')
            self.config_file = Path(self.cache_directory) / 'config.pickle'
            self._cache = dc.Cache(
                self.cache_directory,
                size_limit=self.max_cache,
                eviction_policy=eviction_policy
            )
        else:
            self._store_on_disk = False
            self.cache_directory = None

            # Memory cache
            maxsize = max(1, self.max_cache // 1024)

            if self.lifetime > 0:
                # TTL Cache
                ttl_seconds = self.lifetime * 86400
                self._cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
            else:
                # Cache witout expiration
                if eviction_policy == 'least-recently-used':
                    self._cache = LRUCache(maxsize=maxsize)
                elif eviction_policy == 'least-frequently-used':
                    self._cache = LFUCache(maxsize=maxsize)
                else:  # least-recently-stored o default
                    self._cache = LRUCache(maxsize=maxsize)

            # Thread safety
            self._cache_lock = threading.RLock()

        if hashable_kw is None:
            hashable_kw = []
        self.hashable_kw = hashable_kw

        if verbose:
            print(f"Cache for {self.__class__.__name__} initialized.")
            cache_type = "diskcache" if self._store_on_disk else "cachetools (memory)"
            print(f"Using {cache_type} implementation")
            print(f"Cache initialized with max size: {self.max_cache} bytes")
            print(f"Cache lifetime: {self.lifetime} days")
            print(f"Cache eviction policy: {self.eviction_policy}")
            print(
                f"Cache directory: {self.cache_directory if self._store_on_disk else 'In-memory'}")
            print(f"Hashable keyword arguments: {self.hashable_kw}")
            print(f"Force bypass cache keyword: {self.force}")

    def _create_cache_key(self, hashable_first_argument, **kwargs):
        """
        Creates a cache key based on the first argument and hashable keyword arguments."""
        try:
            # Usar hashkey de cachetools para crear claves consistentes
            compound_data = [hashable_first_argument] + \
                [kwargs[kw] for kw in self.hashable_kw]
            return hashkey(*compound_data)
        except TypeError:
            raise TypeError(
                f"Non-hashable argument detected: {hashable_first_argument}")

    def _get_from_cache(self, key):
        """Obtains value from cache (memory or disk)"""
        if self._store_on_disk:
            return self._cache.get(key)
        else:
            with self._cache_lock:
                return self._cache.get(key)

    def _set_to_cache(self, key, value):
        """Establishes value in cache (memory or disk)"""
        if self._store_on_disk:
            expire_seconds = self.lifetime * 86400 if self.lifetime else None
            self._cache.set(key, value, expire=expire_seconds)
        else:
            with self._cache_lock:
                self._cache[key] = value

    def _is_in_cache(self, key):
        """Verifies if key is in cache (memory or disk)"""
        if self._store_on_disk:
            return key in self._cache
        else:
            with self._cache_lock:
                return key in self._cache

    @property
    def _contents(self):
        """Returns the contents of the cache."""
        if self._store_on_disk:
            for key in self._cache.iterkeys():
                print(f"Key: {key}, Value: {self._cache[key]}")
        else:
            with self._cache_lock:
                for key, value in self._cache.items():
                    print(f"Key: {key}, Value: {value}")

    @property
    def _keys(self):
        """Returns the keys of the cache."""
        if self._store_on_disk:
            return list(self._cache.iterkeys())
        else:
            with self._cache_lock:
                return list(self._cache.keys())

    def clear(self):
        return self._cache.clear()

    def __call__(self, method):
        """Cache main decorator"""

        if self._store_on_disk:
            # For disk, use the diskcache decorator
            @self._cache.memoize(expire=self.lifetime*86400 if self.lifetime else None)
            def cached_method(*args, **kwargs):
                return method(*args, **kwargs)

            def wrapper(hashable_first_argument, **kwargs):
                verbose_func = kwargs.pop('verbose', None)
                verbose = verbose_func if verbose_func is not None else self.verbose

                force_function_arg = kwargs.pop('force', None)
                force_cache = self.force
                force = force_function_arg if force_function_arg is not None else force_cache

                if force:
                    if verbose:
                        logger.info(
                            f"Cache bypassed for method {method.__name__}")
                    return method(hashable_first_argument, **kwargs)

                try:
                    compound_hash = tuple([hashable_first_argument] +
                                          [kwargs[kw] for kw in self.hashable_kw])
                    hash(compound_hash)
                except TypeError:
                    raise TypeError(
                        f"Non-hashable argument detected: {hashable_first_argument}")

                return cached_method(hashable_first_argument, **kwargs)
        else:
            # For memory cache, use cachetools.
            def wrapper(hashable_first_argument, **kwargs):
                verbose_func = kwargs.pop('verbose', None)
                verbose = verbose_func if verbose_func is not None else self.verbose

                force_function_arg = kwargs.pop('force', None)
                force_cache = self.force
                force = force_function_arg if force_function_arg is not None else force_cache

                if force:
                    if verbose:
                        logger.info(
                            f"Cache bypassed for method {method.__name__}")
                    return method(hashable_first_argument, **kwargs)

                # Create cache key.
                try:
                    cache_key = self._create_cache_key(
                        hashable_first_argument, **kwargs)
                except TypeError as e:
                    if verbose:
                        logger.warning(
                            f"Non-hashable arguments, bypassing cache: {e}")
                    return method(hashable_first_argument, **kwargs)

                # Verify cache
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    if verbose:
                        logger.info(f"Cache hit for method {method.__name__}")
                    return cached_result

                # Cache miss
                if verbose:
                    logger.info(f"Cache miss for method {method.__name__}")

                result = method(hashable_first_argument, **kwargs)

                try:
                    self._set_to_cache(cache_key, result)
                    if verbose:
                        logger.info(
                            f"Result cached for method {method.__name__}")
                except Exception as e:
                    if verbose:
                        logger.warning(f"Failed to cache result: {e}")

                return result

        return wrapper

    def clear(self):
        """Cleans the cache"""
        if self._store_on_disk:
            self._cache.clear()
        else:
            with self._cache_lock:
                self._cache.clear()

    def get_stats(self):
        """Obtains cache statistics"""
        if self._store_on_disk:
            return {
                'type': 'diskcache',
                'directory': self.cache_directory,
                'keys_count': len(list(self._cache.iterkeys())),
                'eviction_policy': self.eviction_policy
            }
        else:
            with self._cache_lock:
                return {
                    'type': 'cachetools (memory)',
                    'keys_count': len(self._cache),
                    'maxsize': self._cache.maxsize,
                    'currsize': self._cache.currsize if hasattr(self._cache, 'currsize') else len(self._cache),
                    'cache_class': self._cache.__class__.__name__,
                    'eviction_policy': self.eviction_policy,
                }
