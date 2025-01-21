__all__ = ['CalibBase']


class CalibBase:
    def __init__(self, **kwargs):
        pass

    def short(self):
        return ""

    def __str__(self):
        return "AstroCalib"

    def __call__(self, astrofile, data):
        return data

