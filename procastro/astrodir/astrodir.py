import warnings
import glob
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['AstroDir']

import procastro as pa
from procastro.logging import io_logger
from procastro.statics import dict_from_pattern, glob_from_pattern


class AstroDir:
    def __new__(cls, *args, **kwargs):
        """
        If passed an AstroDir, then do not create a new instance, just pass
        that one
        """
        if args and isinstance(args[0], AstroDir) and len(kwargs) == 0:
            return args[0]

        return super(AstroDir, cls).__new__(cls)

    def __init__(self,
                 files,
                 directory: str | Path = None,
                 spectral=False,
                 **kwargs):
        """

        Parameters
        ----------
        files: str, list
          if str them it is a glob pattern, otherwise a list of specific files
        directory: str, Path
          if None, then files
        """
        if directory is None:
            directory = ""
        directory = Path(directory)
        self.directory: Path = directory
        self._last_sort_key = None
        self.spectral = spectral

        meta_from_name = None
        original_filename = files
        if isinstance(files, str):
            if 0 < files.count('{') == files.count('}'):
                meta_from_name = files
                files = glob_from_pattern(files)

            files = glob.glob(str(directory / files))
            if not len(files):
                io_logger.warning(f"No files found with pattern '{original_filename}'")
        elif isinstance(files, pa.AstroFile):
            files = [files]

        astro_files = []
        for file in files:
            if isinstance(file, AstroDir):
                for astro_file in file:
                    astro_files.append(astro_file)
                continue

            if isinstance(file, (str, Path)):
                astro_file = pa.AstroFile(str(directory / file),
                                          meta_from_name=meta_from_name,
                                          spectral=self.spectral,
                                          **kwargs)
            elif isinstance(file, pa.AstroFile):
                astro_file = file
            else:
                raise TypeError(f"Unrecognized file specification: {file}")
            astro_files.append(astro_file)
        self.astro_files = astro_files

    def sort_key(self, *args):
        """

        Parameters
        ----------
        args:
          many alternatives can be given, but only the first one to be available for every AstroFile is chosen.
        """
        for key in args:
            if None not in self.values(key):
                break
        else:
            raise ValueError("No valid key given that is available for all the AstroFiles")

        for af in self:
            af.add_sortkey(key)
        self._last_sort_key = key

    def sort(self, *args):
        """
        Sorts AstroFile instances inplace depending on the given header fields.
        After sorting the contents the method will return None to avoid problems

        Parameters
        ----------
        args : str
            If given, it first runs sort_key(*args)

        Raises
        ------
        ValueError
            If the field given cannot be used to compare each AstroFile or
            if no header field was specified.

        """
        if len(args):
            self.sort_key(*args)

        self.astro_files.sort()

        return None

    @property
    def filename(self):
        return ', '.join([Path(af.filename).name for af in self])

    def __repr__(self):
        return "<AstroFile container: {0:s}>".format(self.filename)

    def __getitem__(self, item):

        # if an integer, return that AstroFile
        if isinstance(item, (int, np.integer)):
            return self.astro_files[item]  # .__getitem__(item)

        # if string, return as values()
        if isinstance(item, str):
            return self.values(item)

        if isinstance(item, np.ndarray):  # return will be an AstroDir
            if item.dtype == 'bool':  # imitate indexing on boolean array as in scipy.
                if len(item) != len(self):
                    raise ValueError("Attempted to index AstroDir with "
                                     "a boolean array of different size"
                                     "(it must include all bads)")

                astro_files = [f for b, f in zip(item, self) if b]
            elif item.dtype == 'int':  # picks the indexed elements
                astro_files = [self[i] for i in item]
            else:
                raise TypeError(f"Only np.ndarray of type int or bool can be used for indexing, not {item.dtype}")
            return AstroDir(astro_files, directory=self.directory)

        # if it is a slice, return a new astrodir
        if isinstance(item, slice):
            return AstroDir(self.astro_files[item], directory=self.directory)

        # else assume list of integers, then return an Astrodir with those indices
        try:
            return AstroDir([self[i] for i in item])
        except TypeError:
            pass

        raise TypeError(f"item ({item}) is not of a valid type: np.array of booleans, int, str, iterable of ints")

    def __len__(self):
        return len(self.astro_files)

    def filter(self, *args, **kwargs):
        """
        Filter files according to those whose filter return True to the given
        arguments.
        What the filter does is type-dependent for each file.

        Logical 'and' statements can be simulated by chaining this method
        multiple times.

        Parameters
        ----------
        **kwargs :
            Keyword arguments containing the name of the item and the expected
            value.

        Returns
        -------
        procastro.AstroDir
            Copy containing the files which were not discarded.

        See Also
        --------
        procastro.AstroFile.filter :
            Specifies the syntax used by the recieved arguments.

        """

        return AstroDir([f for f in self if f.filter(*args, **kwargs)])

    def values(self, *args, cast=None, by_values=False, single_in_list=False):
        """
        Gets the header values specified in 'args' from each file.
        A function can be specified to be used over the returned values for
        casting purposes.

        Parameters
        ----------
        single_in_list
        cast : function, optional
            Function output used to cas the output
        by_values: bool
          If True, then return an array of size (n_files x n_values)
          If False, then return an array of size (n_values x n_files)

        Returns
        -------
        numpy array with the values.

        """

        if cast is None:
            def cast(x): return x

        warnings.filterwarnings("once",
                                "non-standard convention",
                                AstropyUserWarning)
        ret = np.array([f.values(*args, cast=cast, single_in_list=single_in_list)
                        for f in self])

        warnings.resetwarnings()

        if by_values:
            ret = ret.transpose()

        return ret

    def iter_by(self, *keys, combine=None):
        values = self.values(*keys, single_in_list=True)
        content = values.transpose().tolist() + [list(range(len(self)))]
        table = Table(content, names=list(keys) + ['idx'])

        if len(table) == 0:
            return

        groups = table.group_by(keys).groups
        for group in groups:
            ret = AstroDir(self[group['idx'].data], directory=self.directory)
            if combine is not None:
                ret = combine(ret)

            yield ret

    def _combine_by(self, *keys, combinator=None, in_place=True) -> "AstroDir":

        astro_files = []
        for astro_file in self.iter_by(*keys, combine=combinator):
            astro_files.append(astro_file)
            pass

#        astro_files = [astro_file for astro_file in self.iter_by(*keys, combine=combinator)]

        if in_place:
            self.astro_files = astro_files
            ret = self
        else:
            ret = AstroDir(astro_files, directory=self.directory)

        return ret

    def timeseries_by(self, *keys, in_place=True):

        return self._combine_by(*keys, combinator=pa.AstroFileTimeSeries, in_place=in_place)

    def mosaic_by(self, *keys, in_place=True):

        return self._combine_by(*keys, combinator=pa.AstroFileMosaic, in_place=in_place)

    def __iter__(self):
        self._idx = -1
        return self

    def __next__(self):
        if self._idx >= len(self) - 1:
            raise StopIteration
        self._idx += 1

        return self[self._idx]

    def __add__(self, other):
        sp_img = ['imaging', 'spectral']
        spectral = self.spectral
        directory = self.directory

        if isinstance(other, pa.AstroDir):
            if other.spectral != self.spectral:
                spectral = None
                io_logger.warning(f"Warning, mixing spectral and not spectral AstroDir: "
                                  f"{sp_img[self.spectral]} + {sp_img[other.spectral]}.")
            if other.directory != self.directory:
                io_logger.warning(f"Warning, mixing AstroDir with different directories, "
                                  f"using: {directory}")

            astrofiles = [af for af in other]

        elif isinstance(other, pa.AstroFile):
            astrofiles = [other]

        else:
            raise TypeError("Can only add AstroFile or AstroDir to AstroDir")

        return pa.AstroDir(self.astro_files + astrofiles,
                           spectral=spectral,
                           directory=directory)
