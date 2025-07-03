from procastro.axis.axis import AstroAxis

__all__ = ['SpectralWavAxis']

class SpectralWavAxis(AstroAxis):
    acronym = "W"

    def short(self):
        return f"Wavelength"