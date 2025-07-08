from procastro.astrofile.astrofile import AstroFile

__all__ = ['AstroFileSpec']


class AstroFileSpec(AstroFile):

    def __init__(self,
                 filename,
                 **kwargs):
        """

        Parameters
        ----------
        filename: str, np.ndarray
          Filename or np.ndarray to be used as AstroFile afterward
        file_options: dict
           dictionary with extra information for file readding. (colnames: a list with a name for each channel)
        meta: dict
           Initial meta information, anything here will be overwritten by file reading if field matches name.
        """

        super().__init__(filename, **kwargs)

    ########################################################
    #
    # .read() is the function that reads the data that later is accessed by .meta and .data (returning data and
    # storing meta in self) ... this is likely the main function that any subclass will need to edit
    #
    ##################################

    def __len__(self):
        if not self.spectral:
            raise TypeError(f"object of type '{type(self)} only has len() if it is of spectral type")
        return len(self.data)
