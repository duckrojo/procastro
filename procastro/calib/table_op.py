import re

from astropy.table import Table

from procastro.calib.calib import AstroCalib
from procastro.exceptions import ColumnExistsError, ColumnMissingError
from procastro.config import pa_logger
from procastro.statics import PADataReturn, PAMetaReturn


class TableOp(AstroCalib):
    def __init__(self,
                 add: dict = None,
                 modify: dict = None,
                 force: dict = None,
                 ):
        """

        Parameters
        ----------
        add: dict, optional
        Operation to be eval'd to create a new column, use `[label]`
         to use column `label` in `data` table, or `{label}` to use value
         from `meta` in the operation.  Columns must be non-existent
        modify: dict, optional
        As `add` keyword, but for existing columns
        """
        super().__init__()

        self._add_dict = self.add(add)
        self._modify_dict = self.modify(modify)
        self._force_dict = self.force(force)

    def add(self, add_dict):
        if add_dict is None:
            add_dict = {}
        self._add_dict = add_dict

        return self._add_dict

    def force(self, force_dict):
        if force_dict is None:
            force_dict = {}
        self._force_dict = force_dict

        return self._force_dict

    def modify(self, modify_dict):
        if modify_dict is None:
            modify_dict = {}
        self._modify_dict = modify_dict

        return self._modify_dict

    def _add_modify(self,
                    data,
                    meta,
                    unique):
        for column, value in self._add_dict.items():
            if unique is not None:
                if unique and column in data.colnames:
                    raise ColumnExistsError(f"add operations cannot overwrite existing columns ('{column}'), "
                                            f"maybe you need modify()")
                if not (unique or column in data.colnames):
                    raise ColumnMissingError(f"modify operations cannot overwrite existing columns ('{column}'), "
                                             f"maybe you need modify()")

            operation = re.sub(r"\[(\w+?)]", r"data['\1']", value)
            operation = re.sub(r"\{(\w+?)}", r"meta['\1']", operation)

            pa_logger.debug(f"operation: '{value}' -> \n{operation}")
            try:
                data[column] = eval(operation)
            except Exception as e:
                e.add_note("Error while evaluating: {operation}")
                e.add_note("translated from: {value}")
                raise

        return data, meta

    def _add(self,
             data: Table,
             meta: PAMetaReturn,
             ) -> (PADataReturn, PAMetaReturn):
        self._add_modify(data, meta, True)
        return data, meta

    def _modify(self,
                data: Table,
                meta: PAMetaReturn,
                ) -> (PADataReturn, PAMetaReturn):
        self._add_modify(data, meta, False)
        return data, meta

    def _force(self,
               data: Table,
               meta: PAMetaReturn,
               ) -> (PADataReturn, PAMetaReturn):
        self._add_modify(data, meta, None)
        return data, meta

    def __call__(self, data, meta):
        data, meta = super().__call__(data, meta)
        if self._add_dict is not None:
            data, meta = self._add(data, meta)

        if self._modify_dict is not None:
            data, meta = self._modify(data, meta)

        if self._force_dict is not None:
            data, meta = self._force(data, meta)

        return data, meta

    def short(self):
        return "TableOp"

    def __repr__(self):
        ops = []

        n_modify = len(self._modify_dict)
        if n_modify > 0:
            ops.append(f"add {n_modify} column{'s' if n_modify > 1 else ''} "
                       f"({",".join(self._modify_dict.keys())})")

        n_add = len(self._add_dict)
        if n_add > 0:
            ops.append(f"add {n_add} column{'s' if n_add>1 else ''} "
                       f"({",".join(self._add_dict.keys())})")

        n_force = len(self._force_dict)
        if n_force > 0:
            ops.append(f"add {n_force} column{'s' if n_force>1 else ''} "
                       f"({",".join(self._force_dict.keys())})")

        if not len(ops):
            ops.append("no operations")

        return f"<{super().__repr__()} Table Operation. {". ".join(ops)}>"
