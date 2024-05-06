from typing import Optional

import queue

__all__ = ['astrofile_cache', 'jpl_cache']


class _AstroCache:
    def __init__(self, max_cache=200):
        self._cache: dict = {}
        self._max_cache: int = max_cache
        self._queue: Optional[queue.Queue] = None

        self.set_max_cache(self._max_cache)

    def __bool__(self):
        return self._max_cache > 0

    def available(self):
        return not self._queue.full()

    def __call__(self, method):
        def wrapper(hashable_first_argument, **kwargs):
            """Instance is just the first argument to the function which would need a __hash__:
             in a method it refers to self."""
            cache = True
            try:
                if hashable_first_argument in self._cache:
                    return self._cache[hashable_first_argument]
            except TypeError:
                # disable cache if type is not hashable (numpy array for instance)
                cache = False

            ret = method(hashable_first_argument, **kwargs)

            # if caching
            if cache and self._queue is not None:

                # delete oldest cache if limit reached
                if self._queue.full():
                    del self._cache[self._queue.get_nowait()]

                self._queue.put_nowait(hashable_first_argument)
                self._cache[hashable_first_argument] = ret.copy()

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
