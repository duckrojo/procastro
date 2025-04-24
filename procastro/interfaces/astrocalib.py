import abc


class IAstroCalib(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, data, meta):
        raise NotImplementedError
