from pathlib import Path

from procastro.logging import io_logger
from procastro.statics import identity, PADataReturn
from .spec import AstroFileSpec
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

    def __hash__(self):
        return hash(tuple(self))

    @property
    def name(self):
        ret = f"{self.prefix}({", ".join([str(v.name) for v in self])})"
        return ret

    def __str__(self):
        return str(self[0])


class AstroFileMulti(AstroFileSpec):
    _initialized = False

    @property
    def filename(self):
        return self._data_file.name

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
        self._spectral = spectral
        self.singles = [AstroFileSpec(af, spectral=spectral, **kwargs) for af in astrofiles]

        super().__init__(astrofiles[0], spectral=spectral, **kwargs)

        self._data_file = _FileNames(astrofiles, prefix=self.id_letter)

    def add_calib(self, astrocalibs):
        if astrocalibs is None:
            return self

        if not isinstance(astrocalibs, list):
            astrocalibs = [astrocalibs]

        already_calib = self.get_calib()
        for calib in astrocalibs:
            if calib in already_calib:
                io_logger.warning(f"AstroCalib instance {calib} is being used both in"
                                  f" multifile instance and individual file, thus will be applied twice."
                                  f" Are you sure?")

        return super().add_calib(astrocalibs)

    def get_calib(self) -> tuple:

        ret = tuple([tuple(self._calib)]
                    + [tuple(single.get_calib()) for single in self.singles])
        return ret

    def read(self) -> PADataReturn:
        # in subclasses, do not forget to return data, reset._random, and save ._meta as CaseInsensitiveDict
        #    self._meta = CaseInsensitiveMeta(meta)
        #    self._random = random()
        #
        #    return ret
        raise NotImplementedError("read() must be implemented by subclass")

    @property
    def id_letter(self):
        raise NotImplementedError("This also must be implemented by subclass, ideally just one "
                                  "letter when printing filenames")

    @classmethod
    def get_combinators(cls):
        return cls._combinators
