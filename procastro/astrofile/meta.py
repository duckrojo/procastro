# Based on code from m000
#
# https://stackoverflow.com/a/32888599

from astropy.io import fits as pf


class CaseInsensitiveMeta(dict):
    _commentary = ['history', 'comment', '', 'HISTORY', "COMMENT"]

    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str) else key

    def __init__(self, dict_in):
        dict_in = dict_in.copy()

        to_change = []
        for commentary in self._commentary:
            if commentary in dict_in.keys():
                to_change.append(commentary.lower())

        for commentary in set(to_change):
            out = dict_in.pop(commentary, [])
            if isinstance(out, str):
                out = [out]
            else:
                try:  # if coming from a Header
                    out = list(out)
                except TypeError:
                    out = [out]
            dict_in[commentary] = out

        super(CaseInsensitiveMeta, self).__init__(dict_in)

        self._convert_keys()

    def to_header(self):
        multi_options = {}
        for store in self._commentary:
            multi_options[store] = self.pop(store, [])

        for k, v in self.items():
            if isinstance(v, list):
                self[k] = str(v)
        header = pf.Header(self)

        for option, vals in multi_options.items():
            for val in vals:
                header[option] = val

        return header

    def __getitem__(self, key):
        return super(CaseInsensitiveMeta, self).__getitem__(self.__class__._k(key))

    def __setitem__(self, key, value):
        if key in self._commentary:
            if key in self:
                self[key].append(value)
            else:
                if not isinstance(value, list):
                    value = [value]
                super(CaseInsensitiveMeta, self).__setitem__(self.__class__._k(key), value)
        else:
            super(CaseInsensitiveMeta, self).__setitem__(self.__class__._k(key), value)

    def __delitem__(self, key):
        return super(CaseInsensitiveMeta, self).__delitem__(self.__class__._k(key))

    def __contains__(self, key):
        return super(CaseInsensitiveMeta, self).__contains__(self.__class__._k(key))

    def has_key(self, key):
        return super(CaseInsensitiveMeta, self).has_key(self.__class__._k(key))

    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveMeta, self).pop(self.__class__._k(key), *args, **kwargs)

    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveMeta, self).get(self.__class__._k(key), *args, **kwargs)

    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveMeta, self).setdefault(self.__class__._k(key), *args, **kwargs)

    def update(self, E={}, **F):
        super(CaseInsensitiveMeta, self).update(self.__class__(E))
        super(CaseInsensitiveMeta, self).update(self.__class__(**F))

    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveMeta, self).pop(k)
            self.__setitem__(k, v)

    def __or__(self, other):
        return super().__or__(CaseInsensitiveMeta(other))
