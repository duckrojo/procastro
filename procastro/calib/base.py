
__all__ = ['CalibBase']

from ..core.astrofile import AstroFileBase
from ..core.statics import PADataReturn


class CalibBase:
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        pass

    def short(self):
        return ""

    def __str__(self):
        return "AstroCalib"

    def __call__(self,
                 data: "pa.AstroFile | PADataReturn",
                 meta: dict = None,
                 ) -> tuple[PADataReturn, dict]:

        if meta is None:
            meta = {}

        if isinstance(data, AstroFileBase):
            if self in data.get_calib():
                raise RecursionError(f"If passed an AstroFile, then this calibration cannot have been "
                                     f"already passed to that AstroFile")
            meta = data.meta | meta
            data = data.data

        return data, meta
