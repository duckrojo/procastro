import re

from procastro.axis.axis import AstroAxis

__all__ = ['AstroAxes']


class AstroAxes:
    def __init__(self,
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
        self.specification = specification

        specs = re.findall("([A-Z][a-z]?)", specification)

        if lims is None:
            lims = {}

        # Choose the correct child as AstroAxes according to specification and initialize to correct size
        astro_axes = []
        for spec, dim in zip(specs, shape):
            astro_axis_class = AstroAxis.use(spec)
            linear_lims = lims[spec] if spec in lims else None

            astro_axes.append(astro_axis_class(dim, linear_lims))


        self._axes = astro_axes

    def __str__(self):
        shape = []
        ident = []
        for axes in self._axes:
            ident.append(f'{axes.acronym}')
            shape.append(len(axes))

        return f"AstroAxes: [{', '.join(ident)}] ({'x'.join([str(s) for s in shape])})"

    def __len__(self):
        return len(self._axes)

    def __getitem__(self, item):
        """

        Parameters
        ----------
        item: str
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

