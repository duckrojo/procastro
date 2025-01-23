from procastro.misc import functions
from procastro._bases.calib import CalibBase
from procastro.logging import io_logger
from procastro.astrofile.mosaic import AstroFileMosaic
import procastro as pa

__all__ = ['WavSol']


class WavSol (CalibBase):
    def __init__(self,
                 pixwav: "pa.AstroDir | pa.AstroFile",
                 arcs=None,
                 refit=True,
                 group_by='trace',
                 pixwav_function="poly:d4",
                 **kwargs):
        super().__init__(**kwargs)

        # if group_by is None:
        #     group_by = [[]]
        if not isinstance(group_by, list):
            group_by = [group_by]
        self.group_by = group_by

        if isinstance(pixwav, pa.AstroFile):
            pixwav = pa.AstroDir([pixwav])
        if not pixwav[0].spectral:
            raise TypeError("pixwav must have spectral data")

        self.function = {}
        for astrodir in pixwav.iter_by(*group_by):
            table = AstroFileMosaic(astrodir, spectral=True).data
            option = tuple(astrodir.values(*group_by, single_in_list=True)[0])
            if option in self.function:
                io_logger.warning(f"More than one of tables in pixwav ({pixwav}) has identical grouping")
            self.function[option] = functions.use_function(pixwav_function, table['pix'], table['wav'])

        self.pixwav = pixwav

    def __call__(self,
                 data,
                 meta=None,
                 ):
        data, meta = super().__call__(data, meta=meta)

        group_key = tuple([meta[val] for val in self.group_by])
        data['wav'] = self.function[group_key](data['pix'])

        return data, meta
