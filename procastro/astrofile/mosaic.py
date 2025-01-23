from pathlib import Path
from random import random

from astropy.table import Table, vstack

from .static_guess import static_guess_spectral_offset
from procastro.statics import identity
from procastro.astrodir import AstroDir
from .astrofile import AstroFile

__all__ = ['AstroFileMosaic']


class AstroFileMosaic(AstroFile):

    @property
    def filename(self):
        return "(" + ', '.join([Path(df).name for df in self._data_file]) + ")"

    def __repr__(self):
        return (f"<Multi {'Spec' if self.spectral else 'Image'} {len(self._data_file)} files "
                f"{self.filename}>")

    def __new__(cls, *args, **kwargs):
        """
        If passed an AstroFile, then do not create a new instance, just pass
        that one. If passed None, return None
        """
        new_args = len([key for key, val in kwargs.items() if val is not None])
        if args and ((isinstance(args[0], AstroFile) and new_args == 0) or
#                     (isinstance(args[0], AstroDir) and new_args == 0) or
                     args[0] is None):
            return args[0]

        return super().__new__(cls)

    def __init__(self,
                 astrofiles: str | AstroDir | list[AstroFile],
                 offset: dict[str, float] | None = None,
                 offset_key='chip',
                 spectral: bool = None,
                 **kwargs):

        if isinstance(astrofiles, str):
            astrofiles = [astrofiles]

        if spectral is None:
            try:
                spectral = astrofiles[0].spectral
            except AttributeError:
                raise TypeError("If not given AstroFile iterator to AstroFileMosaic, spectral must be specified")
        self.spectral = spectral

        super().__init__(astrofiles[0], spectral=spectral, do_not_read=True, **kwargs)

        ret = []
        for af in astrofiles:
            ret.append(AstroFile(af, spectral=spectral, **kwargs))
            pass
        self.singles = ret
        # self.singles = [AstroFile(af, spectral=spectral, **kwargs) for af in astrofiles]

        self._data_file = tuple([af.filename for af in astrofiles])

        self.offset_key = offset_key.upper()
        self.offset_values = self.update_offset(offset)

        # first read storing in cache
        identity(self.data)

    def update_offset(self, offset):
        if offset is None:
            offset = static_guess_spectral_offset(self.singles[0].meta)

        self.offset_values = {str(k): v for k, v in offset.items()}
        self._random = random()  # forces re-read of cache

        return self.offset_values

    def __getitem__(self, item):
        return self.singles[item]

    def read(self):
        ret = Table()

        for idx, single in enumerate(self.singles):

            new_table = single.data
            new_table.meta |= single.meta
            ret = vstack([ret, new_table])

        self._meta = ret.meta
        self._random = random()

        return ret
