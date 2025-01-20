import warnings
import glob
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from procastro.core import astrofile, astrodir
from procastro.core.astrofile.static_guess import static_guess_type_from_file

astrofile_type = {'img': astrofile.AstroFile,
                  'spec': astrofile.AstroFileSpec,
                  'mspec': astrofile.AstroFileMosaicSpec,
                  }

__all__ = ['AstroDir']


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
                 type_hints=None,
                 bias=None,
                 flat=None,
                 **kwargs):
        """

        Parameters
        ----------
        files: str, list
          if str them it is a glob pattern, otherwise a list of specific files
        directory: str, Path
          if None, then files
        """
        directory = Path(directory)
        self.directory = directory
        self._last_sort_key = None

        if isinstance(files, str):
            files = glob.glob(directory + files)

        if bias is not None or flat is not None:
            calib = pa.CalibRaw2D(bias=bias, flat=flat)
        else:
            calib = None

        astro_files = []
        for file in files:
            if isinstance(file, pa.AstroDir):
                for astro_file in file:
                    astro_files.append(astro_file)
                continue

            if isinstance(file, (str, Path)):
                use_astrofile = astrofile_type[static_guess_type_from_file(file, type_hints)]
                astro_file = use_astrofile(directory / file, **kwargs).add_calib(calib)
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

    def __repr__(self):
        ret = []
        for af in self:
            ret.append(repr(af).split(": ")[1])
        return "<AstroFile container: {0:s}>".format(', '.join(ret),)

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

    def values(self, *args, cast=None, single_in_list=False):
        """
        Gets the header values specified in 'args' from each file.
        A function can be specified to be used over the returned values for
        casting purposes.

        Parameters
        ----------
        single_in_list
        cast : function, optional
            Function output used to cas the output

        Returns
        -------
        List of integers or tuples
            Type depends on whether a single value or multiple tuples were given

        """

        if cast is None:
            def cast(x): return x

        warnings.filterwarnings("once",
                                "non-standard convention",
                                AstropyUserWarning)
        ret = [f.values(*args, cast=cast, single_in_list=single_in_list) for f in self]

        warnings.resetwarnings()
        return np.array(ret)

    def __add__(self, other):
        if isinstance(other, astrodir.AstroDir):
            other_af = [af for af in other]
        else:
            raise NotImplemented("Only AstroDirs can be added together")

        return astrodir.AstroDir(self.astro_files + other_af)

    def mosaic_by(self, *keys):
        values = self.values(*keys)
        table = Table(values.transpose().tolist()+[range(len(self))], names=list(keys) + ['idx'])

        astro_files = []
        for group in table.group_by(keys).groups:
            ref = self[group['idx'].data][0]
            if len(group) > 1:
                if isinstance(ref, astrofile.AstroFile):
                    raise NotImplementedError("AstroFileMosaic needs to be implemented firs")
                    astro_files.append(astrofile.AstroFileMosaic(self[group['idx'].data], calib=ref.get_calib()))
                elif isinstance(ref, astrofile.AstroFileSpec):
                    astro_files.append(astrofile.AstroFileMosaicSpec(self[group['idx'].data], calib=ref.get_calib()))
                else:
                    raise NotImplementedError("Can only group plain AstroFile and AstroFileSpec for now.  The"
                                              " complex subclasses need to implement concatenate first")

            else:
                astro_files.append(ref)

        return astrodir.AstroDir(astro_files, directory=self.directory)


