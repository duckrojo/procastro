import abc


class IAstroAxis(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def acronym(self):
        """This name identifier must be unique.

        One upper case, or one upper followed by one lowercase character
        """
        raise NotImplementedError

    @abc.abstractmethod
    def short(self):
        """short description of the content"""

        raise NotImplementedError

    @abc.abstractmethod
    def resample(self, **kwargs):
        """Resample the axis"""

        raise NotImplementedError

