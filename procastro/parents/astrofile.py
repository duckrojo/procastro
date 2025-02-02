
__all__ = ['AstroFileBase']

from random import random

from procastro.cache.cache import astrofile_cache
from procastro.astrofile.meta import CaseInsensitiveMeta
from procastro.statics import PADataReturn, identity


class AstroFileBase:
    def __init__(self, *args, **kwargs):
        if len(kwargs) > 0 or len(args) > 0:
            raise TypeError(f"Extra unknown arguments {args} or kwargs {kwargs} passed to AstroFileBase")
        self._calib = []
        self._meta = {}
        self._last_processed_meta = None
        self._random = random()

    def get_calib(self):
        return self._calib

    def read(self):
        raise NotImplementedError("Needs to be implemented by subclass")

    @property
    def meta(self):
        """It will always return a copy"""
        if getattr(self, '_last_processed_meta') is None:
            identity(self.data)
        return CaseInsensitiveMeta(self._last_processed_meta)

    @meta.setter
    def meta(self, value):
        self._meta = CaseInsensitiveMeta(value)
        self._random = random()

    @property
    @astrofile_cache
    def data(self) -> PADataReturn:
        """
        Returns the data in AstroFile by calling .read() the first time and then applying calibration,
        but caching afterward until caching update
        """

        data = self.read()
        meta = self._meta

        for calibration in self._calib:
            data, meta = calibration(data, meta)

        self._last_processed_meta = meta
        return data

