__author__ = 'fran'

class GUIinput(object):
    def __init__(self):
        self.paths = {'bias': None, 'dark': None, 'flat': None, 'sci': None}
        self.combine_mode = {'bias': None, 'dark': None, 'flat': None, 'sci': None}

    def set_paths(self, **kwargs):
        for kind, path in kwargs.items():
            self.paths[kind] = path