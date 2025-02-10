from random import random

from astropy.table import Table, vstack

__all__ = ['AstroFileMosaic']

from astropy.utils.metadata import enable_merge_strategies, MergeStrategy

from .meta import CaseInsensitiveMeta
from .multi import AstroFileMulti


class _MergeDifferentToList(MergeStrategy):
    types = ((str, int, float, list),
             (str, int, float, list))

    @classmethod
    def merge(cls, left, right):
        # print(f"merging {left} & {right}")
        if isinstance(left, list):
            if isinstance(right, list):
                return left + right
            else:
                return left + [right]
        if isinstance(right, list):
            if isinstance(left, list):
                return left + right
            else:
                return [left] + right
        if type(left) is type(right):
            if left != right:
                return [left, right]
            else:
                return left

        return super().merge(left, right)


class AstroFileMosaic(AstroFileMulti):
    # def __repr__(self):
    #     return (f"<Mosaic {'Spec' if self.spectral else 'Image'} {len(self.filename)} files "
    #             f"{self.filename}>")
    #
    def __init__(self,
                 astrofiles: "str | AstroDir | list[AstroFile]",
                 spectral: bool = None,
                 add_column=None,
                 **kwargs):

        super().__init__(astrofiles, spectral, **kwargs)
        self._add_column = add_column or []
        if not isinstance(self._add_column, list):
            self._add_column = [self._add_column]

    def read(self):
        ret = Table()

        for idx, single in enumerate(self.singles):

            new_table = single.data
            with enable_merge_strategies(_MergeDifferentToList):
                new_meta = new_table.meta
                new_table.meta = CaseInsensitiveMeta(new_meta) | CaseInsensitiveMeta(single.meta)

                for col in self._add_column:
                    new_table[col] = new_table.meta[col]
                ret = vstack([ret, new_table])

        self._meta = CaseInsensitiveMeta(ret.meta)
        self._random = random()

        return ret

    @classmethod
    def get_combinators(cls):
        return [(1, cls)]

    @property
    def id_letter(self):
        return "M"

