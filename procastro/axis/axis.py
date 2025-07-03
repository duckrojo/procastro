
import numpy as np
from matplotlib import pyplot as plt

import astropy.units as u

__all__ = ['AstroAxis']

class AstroAxis:
    _available_axis = []
    acronym = None
    unit = None  # default unit
    discrete = False  # Whether it is continuous or discrete
    selectable = True  # Whether it should be selectable by .use()

    def __init__(self,
                 nn,
                 linear_lims: tuple[float, float] | None = None,
                 ):
        if linear_lims is None:
            linear_lims = (0, nn-1)

        values = np.linspace(linear_lims[0], linear_lims[1], nn)

        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self):
        return self._values

    def short(self):
        """Ideally, a single word describing the axis will be used for labeling plots"""
        return str(self)

    def label(self):
        label = self.short()
        if self.unit is not None:
            label = f"{label} [{self.unit}]"
        return label

    def __str__(self):
        return f"AstroAxis of acronym '{self.acronym}'"

    def to_unit(self, new_unit):
        """Change axis unit to new_unit"""
        if isinstance(new_unit, u.Unit):
            self.unit = self.unit.to(new_unit)

    def label_plot(self,
                   ax: plt.Axes,
                   direction: str = "x",
                   ):
        """Puts the correct label in plot, including units"""
        label = f"{self.short}"
        if self.unit is not None:
            label = f"{label} [{self.unit}]"
        getattr(ax, f"set_{direction}label")(label)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        if cls.selectable:
            cls._available_axis.append(cls)

    @classmethod
    def is_acronyn(cls, acronyn):
        if cls.acronym is None:
            raise TypeError(f"AstroAxis child '{cls}' at {cls.__file__} did not have acronym property set, "
                            f"it should not have happened")
        return cls.acronym == acronyn

    @classmethod
    def list_acronyms(cls):
        return [axis.acronym for axis in cls._available_axis]

    @classmethod
    def use(cls, acronym):

        for axis in cls._available_axis:
            if axis.is_acronyn(acronym):
                return axis
        else:
            available = cls.list_acronyms()
            mesg = available[0] if len(available) == 1 else (", ".join(available[:-1])
                                                             + f"{',' if len(available) > 2 else ''}"
                                                               f" or {available[-1]}")
            raise NotImplementedError(f"AstroAxis Acronym {acronym} is not available. Choose one of "
                                      f"{mesg}")


if __name__ == "__main__":

    a=AstroAxis()
    class MyAstroAxis(AstroAxis):
        acronym = "x"

    class MyAstroAxis2(AstroAxis):
        acronym = "y"

    print(MyAstroAxis.list_acronyms())
    print(MyAstroAxis())
    print(MyAstroAxis.use("z")())
