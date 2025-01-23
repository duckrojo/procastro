
__all__ = ['AstroFileBase']

from procastro.core.cache import astrofile_cache
from procastro.core.statics import PADataReturn


class AstroFileBase:
    def __init__(self, filename, **kwargs):
        self._calib = []
        self._meta = {}
        pass

    def get_calib(self):
        return self._calib

    def read(self):
        raise NotImplementedError("Needs to be implemented by subclass")

    @property
    def meta(self):
        return {k.upper(): v for k, v in self._meta.items()}

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

