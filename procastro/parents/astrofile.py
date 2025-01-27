
__all__ = ['AstroFileBase']

from procastro.cache.cache import astrofile_cache
from procastro.other.case_insensitivity import CaseInsensitiveDict
from procastro.statics import PADataReturn


class AstroFileBase:
    def __init__(self, *args, **kwargs):
        if len(kwargs) > 0 or len(args) > 0:
            raise TypeError(f"Extra unknown arguments {args} or kwargs {kwargs} passed to AstroFileBase")
        self._calib = []
        self._meta = {}

    def get_calib(self):
        return self._calib

    def read(self):
        raise NotImplementedError("Needs to be implemented by subclass")

    @property
    def meta(self):
        return CaseInsensitiveDict(self._meta)

    @property
    @astrofile_cache
    def data(self) -> PADataReturn:
        """
        Returns the data in AstroFile by calling .read() the first time and then applying calibration,
        but caching afterward until caching update
        """

        data = self.read()
        meta = self.meta

        for calibration in self._calib:
            data, meta = calibration(data, meta)

        return data

