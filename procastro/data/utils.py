import numpy as np
from astropy.io import fits as pf


class DictInitials(dict):
    def __getitem__(self, item):
        short_idx = len(item)
        keys = np.array(list(self.keys()))
        short_idxs = np.array([k[:short_idx] for k in keys])
        cast = type(item)

        if (item == keys).sum() == 1:
            return super().__getitem__(item)

        location = item == short_idxs
        if location.sum() > 1:
            raise IndexError(f"Ambiguous index '{item}'. Specify between: {list([cast(k) for k in keys[location]])}")

        if location.sum() == 0:
            raise IndexError(f"Index {item} not found")

        return super().__getitem__(type(item)(keys[location][0]))


# Based on code from m000
#
# https://stackoverflow.com/a/32888599

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

        dict_out = {}
        for field, value in dict_in.items():
            if field in self._commentary:
                if isinstance(value, str):
                    value = [value]
                else:
                    try:  # if coming from a Header
                        value = list(value)
                    except TypeError:
                        value = [value]
            dict_out[field] = value

        super().__init__(dict_out)

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


if __name__ == '__main__':
    dct = DictInitials(name=1, nom=3, tok=5, token=9)
    dct2 = DictInitials({'name':1, 'nom':3, 'tok':5, 'token':9})

    print(dct['no'])
