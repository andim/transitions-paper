import string, itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def heatmap(df, imshow=True, zlabel='z', ax=None, cax=None, cbarkwargs=dict(), cbarlabelkwargs=dict(), **kwargs):
    """Plot a heat-map of a pivoted data frame with automatic labeling of axes.

    imshow: if True use imshow, otherwise use pcolormesh (needed if afterwards nonlinear scaling applied to axes)
    
    """
    if ax is None:
        ax = plt.gcf().add_subplot(111)
    if imshow:
        im = ax.imshow(df, extent=(min(df.columns), max(df.columns), min(df.index), max(df.index)), aspect='auto', **kwargs)
    else:
        X, Y = np.meshgrid(df.columns, df.index)
        # automatic axis scaling does not work if there are nans
        defaultkwargs = dict(vmin=np.nanmin(df), vmax=np.nanmax(df))
        defaultkwargs.update(kwargs)
        im = ax.pcolormesh(X, Y, df, **defaultkwargs)
        #FIXME: Once color matplotlib colormesh is fixed (PR submitted) the following line should suffice
        #im = ax.pcolormesh(X, Y, df, **kwargs)
    if cax is None:
        cbar = plt.gcf().colorbar(im, ax=ax, **cbarkwargs)
    else:
        cbar = plt.gcf().colorbar(im, cax=cax, **cbarkwargs)
    if zlabel is not None:
        cbar.set_label(zlabel, **cbarlabelkwargs)
    # workaround for pdf/svg export for more smoothness
    # see matplotlib colorbar documentation
    cbar.solids.set_edgecolor("face")
    # lower limit
    ax.set_xlim(min(df.columns), max(df.columns))
    ax.set_ylim(min(df.index), max(df.index))
    ax.set_xlabel(df.columns.name)
    ax.set_ylabel(df.index.name)
    return im, cbar

def label_axes(fig_or_axes, labels=string.uppercase,
               labelstyle=r'{\sf \textbf{%s}}',
               xy=(-0.05, 0.95), xycoords='axes fraction', **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure or Axes to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : Where to put the label units (len=2 tuple of floats)
    xycoords : loc relative to axes, figure, etc.
    kwargs : to be passed to annotate
    """
    # re-use labels rather than stop labeling
    labels = itertools.cycle(labels)
    axes = fig_or_axes.axes if isinstance(fig_or_axes, plt.Figure) else fig_or_axes
    for ax, label in zip(axes, labels):
        ax.annotate(labelstyle % label, xy=xy, xycoords=xycoords,
                    **kwargs)

def despine(ax, spines=['top', 'right']):
    if spines == 'all':
        spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_visible(False)

def latexboldmultiline(text):
    """make a LaTeX multiline text bold in matplotlib"""
    return '\n'.join([r'{\bf %s}' % line for line in text.split()])

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form numlines x (points per line) x 2 (x and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

## adapted from from: http://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
def colorline(x, y, z=None, ax=None, **kwargs):
    """
    Plot a colored line

    x, y: array of coordinates
    z : array of z values of same shape as x,y, (default: linear spaced)
    ax : axis on which to plot (default: plt.gca()))

    other kwargs passed to LineCollection
    """
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    else:
        z = np.asarray(z)
    segments = make_segments(x, y)
    thiskwargs = dict(cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0))
    thiskwargs.update(kwargs)
    lc = LineCollection(segments, array=z, **thiskwargs)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    return lc

## adapted from http://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
def highrespoints(x, y, factor=10):
    """
    Take points listed in two vectors and returns them at higher
    resolution. Creates approximately factor*(len(x)-1) points including
    the original points

    Returns new x,y arrays
    """
    
    # r is the distance spanned between pairs of points
    r = (np.diff(x)**2+np.diff(y)**2)**.5

    # rtot is a cumulative sum of r, it is used to save time
    rtot = [0.0]
    rtot.extend(r.cumsum())

    dr = r.sum()/((len(x)-1)*factor)
    xnew = []
    ynew = []
    # arc length to which to go
    rcurrent = 0.0
    rcount = 0 
    while rcurrent < r.sum():
        x1,x2 = x[rcount],x[rcount+1]
        y1,y2 = y[rcount],y[rcount+1]
        theta = np.arctan2((x2-x1),(y2-y1))
        alpha = rcurrent-rtot[rcount] 
        rx = np.sin(theta)*alpha+x1
        ry = np.cos(theta)*alpha+y1
        xnew.append(rx)
        ynew.append(ry)
        rcurrent += dr
        if rcount >= len(rtot)-1:
            break
        if rcurrent > rtot[rcount+1]:
            rcurrent = rtot[rcount+1]
            rcount += 1
    xnew.append(x[-1])
    ynew.append(y[-1])

    return np.asarray(xnew), np.asarray(ynew)
