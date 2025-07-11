from procastro.axis.axis import AstroAxis

__all__ = ['DataAxis']

class DataAxis(AstroAxis):
    acronym = "F"
    selectable = False

    def short(self):
        return self._label

    def __init__(self,
                 data,
                 label=None,
                 ):

        # there is no linearly interpolated axis for the data Axes
        super().__init__(1)

        self._values = data
        self._label = label
