import abc

from procastro.interfaces import IAstroFile


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
