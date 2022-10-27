"""
Contains functions to help with plotting.
"""

###########################################
### Imports
###########################################

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as cm
import matplotlib as mpl
import matplotlib.transforms as mtransforms

###########################################
### Generally Helpul Plotting Functions
###########################################

def init_2d_fax(nrows=1, ncols=1, fraction=1.0, labels=True):
    size = set_size('thesis', fraction=fraction, subplots=(nrows, ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
    if (nrows * ncols > 1):
        for ax in axes.flatten():
            ax.grid()
    else:
        axes.grid()

    # Label subfigures by default
    if labels and (nrows * ncols > 1):
        for i, ax in enumerate(axes.flatten()):
            # label physical distance in and down:
            label = list(string.ascii_lowercase)[i] +')'
            trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans, verticalalignment='top',
                    bbox=dict(facecolor='0.7', edgecolor='None', pad=3.0))

    return fig, axes

def ensure_directory_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
        print(f'Made directory at:\n\n{d}')
    return

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 483.69
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
