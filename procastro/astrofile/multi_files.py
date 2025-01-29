from pathlib import Path

from procastro.logging import io_logger
from procastro.statics import identity
from .astrofile import AstroFile
import procastro as pa


class _FileNames(list):
    def __init__(self, astrofiles, prefix=""):
        filenames = []
        for single in astrofiles:
            filename = single.filename
            if isinstance(filename, str):
                filename = Path(filename)
            elif not isinstance(filename, (Path, _FileNames)):
                raise TypeError("filename must be str, Path, or the filename of another astrofile")
            filenames.append(filename)

        super().__init__(filenames)
        self.prefix = prefix

    @property
    def name(self):
        return str(self)

    def __str__(self):
        ret = f"{self.prefix}({", ".join([str(v.name) for v in self])})"
        return ret


class AstroFileMulti(AstroFile):
    _initialized = False

    @property
    def filename(self):
        return self._data_file

    def __repr__(self):
        return "AstroFileMulti. Better description should have been in subclass"

    def __init__(self,
                 astrofiles: "str | AstroDir | list[AstroFile]",
                 spectral: bool = None,
                 **kwargs):
        # The initialized check at super() is not enough since it will be calling AstroFile before
        # super() thus entering in an infinite loop if called with its subclass
        if self._initialized:
            return

        astrofiles = pa.AstroDir(astrofiles)

        if spectral is None:
            try:
                spectral = astrofiles[0].spectral
            except AttributeError:
                raise TypeError("If not given AstroFile iterator to AstroFileMosaic, spectral must be specified")
        self.spectral = spectral

        self.singles = [AstroFile(af, spectral=spectral, **kwargs) for af in astrofiles]

        super().__init__(astrofiles[0], spectral=spectral, do_not_read=True, **kwargs)

        self._data_file = _FileNames(astrofiles, prefix=self.id_letter)

        # first read storing in cache
        identity(self.data)

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
        # in new implementations, do not forget to return data and save ._meta as CaseInsensitiveDict
        raise NotImplementedError("read() must be implemented by subclass")

    @property
    def id_letter(self):
        raise NotImplementedError("This also must be implemented by subclass, ideally just one "
                                  "letter when printing filenames")
