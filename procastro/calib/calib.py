
import astropy.time as apt

__all__ = ['CalibBase']

from procastro.astrofile.astrofile import AstroFileBase
from procastro.interfaces import IAstroCalib
from procastro.statics import PADataReturn


class CalibBase(IAstroCalib):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        pass

    def short(self):
        return ""

    def __str__(self):
        return "AstroCalib"

    def __repr__(self):
        return f"{str(self)}"

    def __call__(self,
                 data: "pa.AstroFile | PADataReturn",
                 meta: dict = None,
                 ) -> tuple[PADataReturn, dict]:

        if meta is None:
            meta = {}

        meta['history'] = f"processed by '{self.short()}' on {apt.Time.now().isot}"

        if isinstance(data, AstroFileBase):
            if self in data.get_calib():
                raise RecursionError(f"If passed an AstroFile, then this calibration cannot have been "
                                     f"already passed to that AstroFile")
            meta = data.meta | meta
            data = data.data

        return data, meta
