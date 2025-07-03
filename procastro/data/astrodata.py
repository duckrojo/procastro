import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from procastro.axis.axes import AstroAxes
from procastro.axis.data import DataAxis


class AstroData:
    unit = None

    def __str__(self):
        return f'AstroData cube{ f'({self.unit})' if self.unit is not None else ''}. {self._axes}'

    def __init__(self,
                 data=None,
                 meta=None,
                 axes: AstroAxes|None = None,
                 label: str = None,
                 ):
        if data is not None:
            self._data = data
            if meta is None:
                raise ValueError("Meta needs to be specified if data is given")
        if meta is not None:
            self._meta = data

        self._axes: AstroAxes | None = axes
        self._label = label

    def plot(self,
             x_var: str,
             y_var: str,
             ax: Axes = None,
             show=True,
             ):

        if ax is None:
            f, ax = plt.subplots()

        if x_var == 'N' or x_var == 'F':
            x_axis = DataAxis(self.data, label=self._label)
        else:
            x_axis = self.axes[x_var]

        if y_var == 'N' or y_var == 'F':
            y_axis = DataAxis(self.data, label=self._label)
        else:
            y_axis = self.axes[x_var]


        ax.plot(x_axis.values, y_axis.values)
        ax.set_xlabel(x_axis.short())
        ax.set_ylabel(y_axis.short())

        if show:
            ax.figure.show()

    @classmethod
    def from_array(cls,
                   data: np.ndarray,
                   meta: dict = None,
                   axes: str = None,
                   label: str = None,
                   ) -> "AstroData":
        """

        Parameters
        ----------
        data
        meta
        axes: str
           String specifying axes types

        Returns
        -------

        """
        if axes is None:
            match len(data.shape):
                case 1:
                    axes = "X"
                case 2:
                    axes = "XY"
                case _:
                    raise ValueError("Axes type needs to be specified explicitly if more than 2 axes")

        axes_list = AstroAxes(axes, data.shape)
        if len(axes_list) != len(data.shape):
            raise ValueError(f"As many axes ({len(axes_list)}) as data dimensions "
                             f"must be specified ({len(data.shape)})")

        if meta is None:
            meta = {}
        return cls(data=data, meta=meta, axes=axes_list, label=label)

    @property
    def meta(self):
        return self._meta

    @property
    def data(self):
        return self._data

    @property
    def axes(self):
        return self._axes

    def __getitem__(self, item):

        return self.data[item]


if __name__ == "__main__":

    arr = np.arange(200)[None, :] * np.ones(400)[:, None]


    ad = AstroData.from_array(arr, label='Flux')
    print(ad[10:40, 50:60])

    ad.plot('X', 'N')
