from random import random

from astropy.table import Table, vstack

from procastro.core.astrofile.spec import AstroFileSpec


__all__ = ['AstroFileMosaicSpec']


class AstroFileMosaicSpec(AstroFileSpec):
    def __init__(self, astrofiles, **kwargs):

        if isinstance(astrofiles, str):
            astrofiles = [astrofiles]

        super().__init__(astrofiles[0], do_not_read=True, **kwargs)

        self._data_file = "multi"
        self.singles = [AstroFileSpec(af, **kwargs) for af in astrofiles]

        self._random = random()

    def read(self):
        ret = Table()
        meta = {}
        for single in self.singles:

            ret = vstack([ret, single.data])
            meta |= single.meta

        self._meta = meta
        self._random = random()

        return ret


if __name__ == "__main__":
    import procastro as pa

    sp = pa.AstroFileMosaicSpec("../../../sample_files/arc.fits")
    sp.plot()
