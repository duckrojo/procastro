from typing import Union, Optional, Tuple

from matplotlib import pyplot as plt

def figaxes(axes: Union[int, plt.Figure, plt.Axes] = None,
            force_new: bool = True,
            clear: bool = True,
            figsize: Optional[Tuple[int, int]] = None,
            nrows: int = 1,
            ncols: int = 1,
            projection=None,
            **kwargs,
            ) -> (plt.Figure, plt.Axes):
    """
    Function that accepts a variety of axes specifications  and returns the output
    ready for use with matplotlib

    Parameters
    ----------
    projection:
        projection to use if new axes needs to be created. If using existing axes this parameter is omitted
    axes : int, plt.Figure, plt.Axes, None
        If axes is None, and multi col/row setup is requested, then it returns an array as in add_subplots().
        Otherwise, it always returns just one Axes instance.
    figsize : (int, int), optional
        Size of figure, only valid for new figures (axes=None)
    force_new : bool, optional
        If true starts a new axes when axes=None (and only then) instead of using last figure
    clear: bool, optional
        Delete previous axes content, if any

    Returns
    -------
    Matplotlib.pyplot figure and axes
    """
    if axes is None:
        if force_new:
            fig, axs = plt.subplots(nrows, ncols,
                                    figsize=figsize,
                                    subplot_kw=dict(projection=projection),
                                    )
        else:
            plt.gcf().clf()
            fig, axs = plt.subplots(nrows, ncols,
                                    figsize=figsize,
                                    num=plt.gcf().number,
                                    subplot_kw=dict(projection=projection),
                                    )
    elif isinstance(axes, int):
        fig = plt.figure(axes, **kwargs)
        if clear or len(fig.axes) == 0:
            fig.clf()
            axs = fig.add_subplot(nrows, ncols, 1,
                                  projection=projection,
                                  )
        else:
            axs = fig.axes[0]
    elif isinstance(axes, plt.Figure):
        fig = axes
        if clear:
            fig.clf()
        if len(fig.axes) == 0:
            fig.add_subplot(nrows, ncols, 1,
                            projection=projection,
                            )
        axs = fig.axes[0]
    elif isinstance(axes, plt.Axes):
        axs = axes
        if clear:
            axs.cla()
        fig = axes.figure
    else:
        raise ValueError("Given value for axes ({0:s}) is not"
                         "recognized".format(axes, ))

    return fig, axs
