import abc
import pathlib

import numpy as np


class IAstroFile(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_calib(self, calib):
        raise NotImplementedError

    @abc.abstractmethod
    def get_calib(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_sortkey(self,
                    key: str):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self):
        raise NotImplementedError

    @abc.abstractmethod
    def filename(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write_as(self,
                 filename: str | pathlib.Path,
                 overwrite: bool = False,
                 data: np.ndarray | None = None,
                 channels: int = None,
                 backup_extension: str = ".bak",
                 ):
        raise NotImplementedError
