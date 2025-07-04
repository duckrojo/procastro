import re

from procastro.axis.axis import AstroAxis

__all__ = ['AstroAxes']


class AstroAxes:
    def __init__(self,
                 astro_axes: list[AstroAxis],
                 ):
        self.specification = "".join([spec.acronym for spec in astro_axes])

        self._axes = astro_axes

    ################################3
    #
    # Initializing

    @classmethod
    def from_linear(cls,
                    specification: str,
                    shape: tuple[int, ...],
                    lims: dict = None,
                    ):
        """

        Parameters
        ----------
        specification
        shape
        lims: dict
           Dictionary containing the (min, max) limits from each axis to initialize them linearly.
        """

        specs = re.findall("([A-Z][a-z]?)", specification)

        if lims is None:
            lims = {}

        # Choose the correct child as AstroAxes according to specification and initialize to correct size
        astro_axes = []
        for spec, dim in zip(specs, shape):
            astro_axis_class = AstroAxis.use(spec)
            linear_lims = lims[spec] if spec in lims else None

            astro_axes.append(astro_axis_class(dim, linear_lims))


        return cls(astro_axes)

    #################################
    #
    # methods

    def str_available(self):
        ident = [f'{axes.acronym}' for axes in self._axes]
        return ", ".join(ident)

    def __str__(self):
        shape = [str(len(axis)) for axis in self._axes]
        shapes = 'x'.join([str(s) for s in shape])

        return f"AstroAxes: [{self.str_available()}] ({shapes})"

    def __len__(self):
        return len(self._axes)

    def __getitem__(self,
                    item: str | int):
        """

        Parameters
        ----------
        item: str, int
          If str, selects the specified axis according to acronym.
          If int, select the indexed axis.

        Returns
        -------

        """

        if isinstance(item, str):
            if len(item) > 2:
                raise ValueError(f"AstroAxis specification '{item}' is not valid")
            elif len(item) == 1:
                item = item.upper()
            elif len(item) == 2:
                item = item[0].upper() + item[1].lower()
            for axis in self._axes:
                if axis.acronym == item:
                    return axis
            else:
                raise ValueError(f"AstroAxis specification '{item}' not found")

        if isinstance(item, int):
            return self._axes[item]

        raise IndexError("Needs to specify either a valid acronym or a valid index for axis")

    def removed(self,
                index: str | int,
                ):
        """Returns a copy of the axes with the indexed AstroAxis removed."""

        if isinstance(index, str):
            index = self.index(index)

        axes = self._axes[:index] + self._axes[index + 1:]

        return AstroAxes(axes)

    def index(self,
              label):
        """
Returns index of labeled axis.

        Parameters
        ----------
        label
        """
        if isinstance(label, int):
            if label > len(self._axes):
                raise ValueError(f"AstroAxis index {label} is beyond available dims ({len(self._axes)}")
            return label

        ret = None
        for i, axis in enumerate(self._axes):
            if axis.acronym == label:
                if ret is not None:
                    raise ValueError(f"AstroAxis index {label} is repeated")
                ret = i

        if ret is None:
            raise ValueError(f"AstroAxis specification '{label}' not found")

        return ret
