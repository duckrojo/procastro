import copy
import os
import pickle
import tempfile
from typing import Optional
import queue
import pandas as pd

__all__ = ['astrofile_cache', 'jpl_cache', 'usgs_map_cache']


from .misc_general import user_confdir
import astropy.time as apt
import astropy.units as u


class _AstroCache:
    def __init__(self,
                 max_cache=200, lifetime=0,
                 hashable_kw=None, label_on_disk=None,
                 force: str | None = None,
                 ):
        """

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
            self.cache_directory = user_confdir(f'cache/{label_on_disk}', use_directory=True)
            self.config_file = user_confdir(f'cache/{label_on_disk}/config.pickle')
            if not os.path.exists(self.config_file):
                pickle.dump({}, open(self.config_file, 'wb'))
            self._cache = pickle.load(open(self.config_file, 'rb'))
        else:
            self._cache: dict[any, tuple[apt.Time, any]] = {}
            self.cache_directory = None
            self.config_file = None

        sorted_hash = [ch
                       for ch, tm
                       in sorted([(h, apt.Time(t))
                                  for h, (t, c)
                                  in self._cache.items()],
                                 key=lambda x: x[1])
                       ]
        self._queue: Optional[queue.Queue] = None
        self.set_max_cache(self._max_cache)
        if self._queue is not None:
            for h in sorted_hash:
                self._queue.put_nowait(h)

        if hashable_kw is None:
            hashable_kw = []
        self._hashable_kw = hashable_kw

    def __bool__(self):
        return self._max_cache > 0

    def available(self):
        return not self._queue.full()

    def _delete_cache(self):
        compound_hash = self._queue.get_nowait()
        if self._store_on_disk:
            os.remove(self._cache[compound_hash][1])
            self._update_config_file()
        del self._cache[compound_hash]

        return compound_hash

    def _store_cache(self, compound_hash, content):
        self._queue.put_nowait(compound_hash)
        if self._store_on_disk:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.cache_directory, delete=False) as fp:
                pickle.dump(content, fp)
                self._cache[compound_hash] = (apt.Time.now().isot, fp.name)
            self._update_config_file()
        else:
            self._cache[compound_hash] = (apt.Time.now().isot, copy.copy(content))

    def _update_config_file(self):
        pickle.dump(self._cache, open(self.config_file, 'wb'))

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
                # expire cache if too old
                if (self.lifetime and
                    apt.Time.now() - apt.Time(self._cache[compound_hash][0],
                                              format='isot', scale='utc') > self.lifetime * u.day):

                    # empty queue of all objects older than the one requested
                    old_compound_hash = self._delete_cache()
                    while old_compound_hash != compound_hash:
                        old_compound_hash = self._delete_cache()

                # use disk or memory cache
                elif self._store_on_disk:
                    return pickle.load(open(self._cache[compound_hash][1], 'rb'))

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

        # if reducing cache, get rid of the extra caches
        if delta > 0:
            for af in range(delta):
                self._delete_cache()

        try:
            # copy old queue into new
            while True:
                self._queue.put_nowait(old_queue.get_nowait())
        except queue.Empty:
            pass


astrofile_cache = _AstroCache()
jpl_cache = _AstroCache(max_cache=50)
usgs_map_cache = _AstroCache(max_cache=30, lifetime=7,
                             hashable_kw=['detail'], label_on_disk='USGSmap',
                             force="no_cache")

