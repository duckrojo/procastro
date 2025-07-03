from procastro.axis.axis import AstroAxis

__all__ = ['SpectralPixAxis']


class SpectralPixAxis(AstroAxis):
    acronym = "Sx"
    unit = None

    def short(self):
        return "Spectral Pixel"

