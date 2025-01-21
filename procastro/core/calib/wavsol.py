from pathlib import Path

import procastro as pa
from procastro.core import functions
from procastro.core.calib import CalibBase
from procastro.core.logging import io_logger


class WavSol (CalibBase):
    def __init__(self,
                 pixwav: pa.AstroDir | pa.AstroFile,
                 arcs=None,
                 refit=True,
                 group_by=None,
                 pixwav_function="poly:4d",
                 **kwargs):
        super().__init__(**kwargs)

        if group_by is None:
            group_by = [[]]
        self.group_by = group_by

        if isinstance(pixwav, pa.AstroFile):
            pixwav = pa.AstroDir([pixwav])
        if not pixwav[0].spectral:
            raise TypeError("pixwav must have spectral data")

        self.function = {}
        for astrodir in pixwav.iter_by(*group_by):
            table = pa.AstroFileMosaic(astrodir, spectral=True).data
            option = tuple(astrodir.values(*group_by, single_in_list=True)[0])
            if option in self.function:
                io_logger.warning(f"More than one of tables in pixwav ({pixwav}) has identical grouping")
            self.function[option] = functions.use_function(pixwav_function, table['pix'], table['wav'])

        self.pixwav = pixwav

    def __call__(self,
                 astrofile: pa.AstroFile,
                 data,
                 ):
        group_key = tuple(astrofile.values(*self.group_by))
        data['wav'] = self.function[group_key](data['pix'])

        return data
