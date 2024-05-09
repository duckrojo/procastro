import copy
import os
import pickle
import tempfile
from typing import Optional
import queue
import pandas as pd

__all__ = ['astrofile_cache', 'jpl_cache']


from .misc_general import user_confdir
import astropy.time as apt


class _AstroCache:
    def __init__(self,
                 max_cache=200, lifetime=0,
                 hashable_kw=None, label_on_disk=None
                 ):
        """

        Parameters
        ----------
        max_cache
         how many days to cache in disk, if more than that has elapsed, it will be reread.
        """
        self._queue: Optional[queue.Queue] = None

        self._max_cache: int = max_cache
        self.set_max_cache(self._max_cache)
        self.lifetime = lifetime

        if label_on_disk is not None:
            if any((not_permitted in label_on_disk) for not_permitted in ('/', ':')):
                raise ValueError(f"label_on_disk contains invalid file characters '{label_on_disk}'")
            self._store_on_disk = True
        else:
            self._store_on_disk = False

        if self._store_on_disk:
            self.cache_directory = user_confdir(f'cache/{label_on_disk}', use_directory=True)
            self.config_file = user_confdir(f'cache/{label_on_disk}/config.pickle')
            if not os.path.exists(self.config_file):
                pickle.dump(self._cache, open(self.config_file, 'wb'))
            self._cache = pd.read_pickle(self.config_file)
        else:
            self._cache: dict[any, tuple[apt.Time, any]] = {}
            self.cache_directory = None
            self.config_file = None

        if hashable_kw is None:
            self.hashable_kw = []

    def __bool__(self):
        return self._max_cache > 0

    def available(self):
        return not self._queue.full()

    def _delete_cache(self):
        compound_hash = self._queue.get_nowait()
        if self._store_on_disk:
            os.remove(self._cache[compound_hash][1])
        del self._cache[compound_hash]

        return compound_hash

    def _store_cache(self, compound_hash, content):
        self._queue.put_nowait(compound_hash)
        if self._store_on_disk:
            with tempfile.NamedTemporaryFile(dir=self.cache_directory, delete=False) as fp:
                fp.write(content)
                self._cache[compound_hash] = (apt.Time.now().isot, fp.name)
        else:
            self._cache[compound_hash] = (apt.Time.now().isot, copy.copy(content))

    def __call__(self, method):
        def wrapper(hashable_first_argument, **kwargs):
            """Instance is just the first argument to the function which would need a __hash__:
             in a method it refers to self."""
            cache = True
            compound_hash = tuple([hashable_first_argument] +
                                  [kwargs[kw] for kw in self.hashable_kw if kw in kwargs])

            try:
                if compound_hash in self._cache:
                    pass
            except TypeError:
                # disable cache if type is not hashable (numpy array for instance)
                cache = False

            if cache and (compound_hash in self._cache):
                # expire cache if too old
                if (self.lifetime and
                    apt.Time.now() - apt.Time(self._cache[compound_hash][0],
                                              format='isot', scale='utc') > self.lifetime):

                    # empty queue of all objects older than the one requested
                    old_compound_hash = self._delete_cache()
                    while old_compound_hash != compound_hash:
                        old_compound_hash = self._delete_cache()

                # use disk or memory cache
                elif self._store_on_disk:
                    return open(self._cache[compound_hash][1], 'rb').read()

                else:
                    return self._cache[compound_hash][1]

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

        # if disabling cache, delete all
        if not max_cache:
            self._queue = None
            self._cache = {}
            return

        delta = self._max_cache - max_cache
        old_queue = self._queue

        self._queue = queue.Queue(maxsize=max_cache)
        self._max_cache = max_cache

        # if cache was disabled, then start it empty
        if old_queue is None:
            return

        try:
            # if reducing cache, get rid of the extra caches
            if delta > 0:
                for af in range(delta):
                    del self._cache[self._queue.get_nowait()]  # both delete elements from indexing and the cache.

            # copy old queue into new
            while True:
                self._queue.put_nowait(old_queue.get_nowait())
        except queue.Empty:
            pass


astrofile_cache = _AstroCache()
jpl_cache = _AstroCache(max_cache=50)
