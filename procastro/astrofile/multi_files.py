from pathlib import Path

from procastro.logging import io_logger
from procastro.statics import identity
from .astrofile import AstroFile


class AstroFileMulti(AstroFile):

    @property
    def filename(self):
        return "(" + ', '.join([Path(df).name for df in self._data_file]) + ")"

    def __repr__(self):
        return "AstroFileMulti. Better description should have been in subclass"

    def __init__(self,
                 astrofiles: "str | AstroDir | list[AstroFile]",
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

        self.singles = [AstroFile(af, spectral=spectral, **kwargs) for af in astrofiles]

        super().__init__(astrofiles[0], spectral=spectral, do_not_read=True, **kwargs)

        self._data_file = tuple([af.filename for af in astrofiles])

        # first read storing in cache
        identity(self.data)

    def __getitem__(self, item):
        return self.singles[item]

    def add_calib(self, astrocalibs):
        if astrocalibs is None:
            return self

        for single in self.singles:
            for calib in single.get_calib():
                position = astrocalibs.index(calib)
                if position is not None:
                    io_logger.warning(f"AstroCalib instance {astrocalibs[position]} is being used both in"
                                      f" multifile instance and individual file, thus will be applied twice."
                                      f" Are you sure?")

        return super().add_calib(astrocalibs)

    def read(self):
        NotImplementedError("read() must be implemented by subclass")
