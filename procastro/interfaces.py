import abc
import pathlib

import numpy as np


class IAstroFile(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_calib(self, AstroCalib):
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
                 ):
        raise NotImplementedError


class IAstroCalib(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, data, meta):
        raise NotImplementedError


class IAstroDir(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self,
                 filename: "IAstroFile|str|pathlib.Path|IAstroDir"):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __next__(self) -> IAstroFile:
        raise NotImplementedError
