from procastro.axis.axis import AstroAxis

__all__ = ['TemporalAxis']

class TemporalAxis(AstroAxis):
    acronym = "T"

    def short(self):
        return f"Time"