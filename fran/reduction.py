__author__ = 'fran'

import CPUmath


class Reduction(object):
    def __init__(self):
        self.auto = False  # Automatically detect BIAS and dark frames or not
        self.paths = {'bias': None, 'dark': None, 'flat': None, 'sci': None}
        self.combine_type = {'bias': 'mean', 'dark': 'mean'}  # Mean, median, linear interpolation, best

    def set_paths(self, **kwargs):
        for kind, path in kwargs.items():
            self.paths[kind] = path
