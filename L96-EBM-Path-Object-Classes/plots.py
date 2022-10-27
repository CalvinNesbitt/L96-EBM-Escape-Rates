"""
Plots of the L96-EBM System that we will commonly look at.

--------------------------------------------------------------------------------
Contents
--------------------------------------------------------------------------------

- Generally Helpul Plotting Functions
- Default Plot Settings
- Bifurcation diagrams
- 3D Projections
- 2D Projections
- Multi-Projection Plots
- Timeseries Plots
- 2D Probabilistic Plots
- 3D Probabilistic Plots
- Probabilistic Section Plots
"""
###########################################
### Imports
###########################################

# Standard Packages
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as cm
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import numpy as np
from scipy import stats

import os
import string

# Custom Code
from observables import *
from pathObject_IO import get_m_state, get_sb_attractor, get_w_attractor, get_smooth_m_state
from personal_stats import kde_density_2d

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

def init_3d_fax():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    return fig, ax

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


def add_attractors(plot, fax, S=10):
    fig, ax = fax
    hot_attractor = get_hot_attractor(S=S)
    cold_attractor = get_cold_attractor(S=S)
    m_state = get_m_state(S=S)
    colors = ['r', 'b', 'g']
    for ds, c in zip([hot_attractor, cold_attractor, m_state], colors):
        plot(ds, c=c, ax=ax)
    return

def rhs_T(avg_energy, T, p):
    """Compute the right hand side of the T-ODE."""

    [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta] = p


    dTdt = (
        S * (1 - a0 + 0.5 * a1 * (np.tanh(T - Tref)))
        - sigma * T**4 - alpha * (avg_energy/(0.6 * F**(4/3)) - 1)
    )
    return dTdt

def T_dot_plot(fax, energy_values=None, temp_values=None, p=None):
    if p is None:
        K = 50
        S = 10
        a0 = 0.5
        a1 = 0.4
        sigma = 1/180**4
        F = 8
        Tref = 270
        delT = 60
        alpha = 2
        beta = 1
        p = [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta]

    no_of_points = 100

    if energy_values is None:
        energy_values = np.linspace(0, 30, no_of_points)

    if temp_values is None:
        temp_values = np.linspace(250, 300, no_of_points)


    E, T = np.meshgrid(energy_values, temp_values)

    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    im = ax.pcolormesh(E, T, rhs_T(E, T, p), cmap='seismic', vmin = -6, vmax=6, shading='auto')

    return

###########################################
### Default plot settings
###########################################
plt.style.use('thesis')
mpl.rcParams.update({'figure.figsize': set_size('thesis', fraction=0.9)})

###########################################
### Bifurcation Plots
###########################################

def bifurcation_diagram_plot(obs, fax=None, **kwargs):
    "Plots bifurcation diagram for specified observable and S_range"

    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    # Getting Attractor Data
    cold_attractors = [get_cold_attractor(S=S) for S in np.arange(7, 16)]
    hot_attractors = [get_hot_attractor(S=S) for S in np.arange(8, 17)]
    m_states = [get_m_state(S=S) for S in  np.arange(8, 16)] # S where we computed M-State

    # Getting observable Means
    cold_means = [obs(ds).mean() for ds in cold_attractors]
    hot_means = [obs(ds).mean() for ds in hot_attractors]
    m_means = [obs(ds).mean() for ds in m_states]

    # Bifurcation Plot
    ax.plot(np.arange(7, 16), cold_means, c='b', label='SB', **kwargs)
    ax.plot(np.arange(8, 17), hot_means, c='r', label='W', **kwargs)
    ax.plot(np.arange(8, 16), m_means, c='g', label='M', **kwargs)

    ax.set_xlabel('S')
    ax.grid()
    ax.legend()

    return fig, ax

def temperature_bifurcation_diagram_plot(fax=None, **kwargs):
    fig, ax = bifurcation_diagram_plot(obs=temperature, fax=fax, **kwargs)
    ax.set_ylabel('$\overline{\mathcal{T}}$')
    return fig, ax

def mean_x_bifurcation_diagram_plot(fax=None, **kwargs):
    fig, ax = bifurcation_diagram_plot(obs=momentum, fax=fax, **kwargs)
    ax.set_ylabel('$\overline{\mathcal{M}}$')
    return fig, ax

def energy_bifurcation_diagram_plot(fax=None, **kwargs):
    fig, ax = bifurcation_diagram_plot(obs=energy, fax=fax, **kwargs)
    ax.set_ylabel('$\overline{\mathcal{E}}$')
    return fig, ax

###########################################
### 3D Projections
###########################################

def EMT_plot(ds, ax=None, *args, **kwargs):
    "Plots energy, momentum, temperature 3d plot of a dataset."
    if ax is None:
        fig, ax = init_3d_fax()
    ax.plot(momentum(ds), energy(ds), ds.T, *args, **kwargs)
    ax.set_ylabel('$\\mathcal{E}$')
    ax.set_xlabel('$\\mathcal{M}$')
    ax.set_zlabel('$T$')
    return

def xjkT_plot(ds, ax=None, j=1, k=2, *args, **kwargs):
    "x_k vs x_j vs. T plot"
    if ax is None:
        fig, ax = init_3d_fax()
    ax.plot(ds.X.sel(space=j), ds.X.sel(space=k), ds.T, *args, **kwargs)
    ax.set_xlabel(f'$X_{j}$')
    ax.set_ylabel(f'$X_{k}$')
    ax.set_zlabel('$T$')
    return

###########################################
### 2D Projections
###########################################

def ET_plot(ds, *args, fax=None,  **kwargs):
    "Plots energy, momentum, temperature 3d plot of a dataset."
    if fax is None:
        fig, ax = init_2d_fax()
        fax = [fig, ax]
    else:
        fig, ax = fax
    ax.plot(energy(ds), ds.T, *args, **kwargs)
    ax.set_xlabel('$\\mathcal{E}$')
    ax.set_ylabel('$\\mathcal{T}$')
    return

def EM_plot(ds, *args, fax=None,  **kwargs):
    "Plots energy, momentum, temperature 3d plot of a dataset."
    if fax is None:
        fig, ax = init_2d_fax()
        fax = [fig, ax]
    else:
        fig, ax = fax
    ax.plot(energy(ds), momentum(ds), *args, **kwargs)
    ax.set_xlabel('$\\mathcal{E}$')
    ax.set_ylabel('$\\mathcal{M}$')
    return

def MT_plot(ds, *args, fax=None,  **kwargs):
    "Plots energy, momentum, temperature 3d plot of a dataset."
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax
    ax.plot(momentum(ds), ds.T, *args, **kwargs)
    ax.set_xlabel('$\\mathcal{M}$')
    ax.set_ylabel('$\\mathcal{T}$')
    return

def xjT_plot(ds, ax=None, j=1 , *args, **kwargs):
    "x_j vs. T plot"
    ax.plot(ds.X.sel(space=j), ds.T, *args, **kwargs)
    ax.set_xlabel(f'$X_{j}$')
    ax.set_ylabel('$T$')
    return

def xjk_plot(ds, ax=None, j=1, k=2, *args, **kwargs):
    "x_k vs x_j vs. T plot"
    ax.plot(ds.X.sel(space=j), ds.X.sel(space=k), *args, **kwargs)
    ax.set_xlabel(f'$X_{j}$')
    ax.set_ylabel(f'$X_{k}$')
    return

def plot_attractors(ax, plot_projection, W=True, SB=True, M=True, S=10):
    "Plots l96 ebm attractors"
    data = [get_cold_attractor(S=S), get_hot_attractor(S=S), get_m_state(S=S)]
    boolean = [SB, W, M]
    colors = ['b', 'r', 'g']
    for i in range(3):
        if boolean[i]:
            ds = data[i]
            c = colors[i]
            plot_projection(ds, ax, c=c)
    return

###########################################
### Multi-Projection Plots
###########################################

def EMT_with_projections(instanton_list, attractor_pair):
    attractor, attractor_color = attractor_pair
    # EMT Projection of all Instantons
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('EMT Projections')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    for instanton in instanton_list:
        EMT_plot(instanton, ax=ax, c='purple')
    EMT_plot(get_m_state(), ax=ax, c='g')
    EMT_plot(attractor, ax=ax, c=attractor_color)

    for i, plot in enumerate([EM_plot, ET_plot, MT_plot]):
        ax = fig.add_subplot(2, 2, i + 2)
        ax.grid()
        for instanton in instanton_list:
            plot(instanton, ax=ax, c='purple')

        plot(get_m_state(), ax=ax, c='g')
        plot(attractor, ax=ax, c=attractor_color)
    plt.show()
    return fig, ax

def mean_EMT_with_projections(instanton_ds, attractor_pair):
    attractor, attractor_color = attractor_pair
    # EMT Projection of all Instantons
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('EMT Projections')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    EMT_plot(mean_instanton, ax=ax, c='purple')
    EMT_plot(get_m_state(), ax=ax, c='g')
    EMT_plot(attractor, ax=ax, c=attractor_color)

    for i, plot in enumerate([EM_plot, ET_plot, MT_plot]):
        ax = fig.add_subplot(2, 2, i + 2)
        ax.grid()
        plot(mean_instanton, ax=ax, c='purple')

        plot(get_m_state(), ax=ax, c='g')
        plot(attractor, ax=ax, c=attractor_color)
    plt.show()
    return

###########################################
### Timeseries Plots
###########################################

def T_timeseries_plot(ds, *args, fax=None,  **kwargs):
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax
    ds.T.plot(ax=ax, *args, **kwargs)
    ax.set_title('Temperature Timeseries')
    ax.set_ylabel('$\\mathcal{T}$')
    return

def E_timeseries_plot(ds, *args, fax=None,  **kwargs):
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax
    energy(ds).plot(ax=ax, *args, **kwargs)
    ax.set_title('Energy Timeseries')
    ax.set_ylabel('$\\mathcal{E}$')
    return

def M_timeseries_plot(ds, *args, fax=None,  **kwargs):
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax
    momentum(ds).plot(ax=ax, *args, **kwargs)
    ax.set_title('$<X_n>$ Timeseries')
    ax.set_ylabel('$<X_n>$')
    return

def EMT_timeseries_plots(instanton_list, instanton_ds):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    names = ['Temperature', 'Energy', 'Mean']
    labels = ['$\\mathcal{T}$', '$\\mathcal{E}$' , '$<X_n>$']
    fig.suptitle('Instanton Timeseries')
    for instanton in instanton_list:
        for i, observable in enumerate([temperature, energy, momentum]):
            observable(instanton).plot(ax=axes[i], c='b')

    for i, observable in enumerate([temperature, energy, momentum]):
        observable(instanton_ds).mean(dim='realisation').plot(ax=axes[i], c='r')
        observable(instanton_ds).quantile(q=0.05, dim='realisation').plot(ax=axes[i], c='orange')
        observable(instanton_ds).quantile(q=0.95, dim='realisation').plot(ax=axes[i], c='orange')

    for i in range(3):
        axes[i].grid()
        axes[i].set_ylabel(labels[i])
        axes[i].set_title(names[i])
    return fig, axes

###########################################
### 1D Probabilistic Plots
###########################################
"Uses sns.kdeplot to do 1d projections"

def kde_plot(x, *args, grid_points=50, fax=None, x_label='', **kwargs):
    "KDE density plot for 1D np.array data x."

    # Compute KDE and Grid
    kde = stats.gaussian_kde(x)
    grid = np.linspace(x.min() - 2*x.std(), x.max() + 2*x.std(), grid_points)

    # Plot figure
    if fax is None:
        fax=init_2d_fax()
    fig, ax = fax
    ax.plot(grid, kde(grid), *args, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel('$ \\rho $')
    ax.set_title(f'{x_label} Density. {len(x)} samples.')
    ax.grid()
    return fax

def obs_1d_density_plot_maker(observable, x_label='', title='Density'):
    "Makes 2D Desnity Plot function using kde_density_2d_plot"
    def obs_1d_density_plot(ds, *args, fax=None,  **kwargs):

        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        sns.kdeplot(observable(ds).values.flatten(), ax=ax, *args, **kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel('$\\rho$')
        ax.grid()
        ax.set_title(title)
        return fig, ax
    return obs_1d_density_plot

E_density_plot = obs_1d_density_plot_maker(energy, x_label='$\\mathcal{E}$', title='Energy Density')
M_density_plot = obs_1d_density_plot_maker(momentum, x_label='$\\mathcal{M}$', title='Mean X Density')
T_density_plot = obs_1d_density_plot_maker(temperature, x_label='$\\mathcal{T}$', title='Temperature Density')

###########################################
### 2D Probabilistic Plots
###########################################

def kde_density_2d_plot(x, y, fax=None, nbins=50, x_lims=None, y_lims=None, color_min = 0, colorbar=True, *args, **kwargs):

    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    xy = np.vstack([x,y])
    density = stats.gaussian_kde(xy)

    # Set Limits of Density Search
    if x_lims is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min, x_max = x_lims

    if y_lims is None:
        y_min = y.min()
        y_max = y.max()
    else:
        y_min, y_max = y_lims


    xi, yi = np.mgrid[x_min:x_max:nbins*1j, y_min:y_max:nbins*1j]
    di = density(np.vstack([xi.flatten(), yi.flatten()]))
    di = np.ma.masked_array(di, di < color_min)

    im = ax.pcolormesh(xi, yi, di.reshape(xi.shape), *args, **kwargs)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    if colorbar:
        plt.sca(ax)
        plt.colorbar(im)

    return fig, ax

def obs_2d_density_plot_maker(observables, names=None):
    "Makes 2D Desnity Plot function using kde_density_2d_plot"
    def obs_2d_density_plot(ds, fax=None, nbins=50, x_lims=None, y_lims=None, color_min=0, colorbar=True, *args, **kwargs):
        x = observables[0](ds).values.flatten()
        y = observables[1](ds).values.flatten()
        fig, ax = kde_density_2d_plot(x, y, fax=fax, nbins=nbins, x_lims=x_lims, y_lims=y_lims, color_min=color_min, colorbar=colorbar, *args, **kwargs)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        return fig, ax
    return obs_2d_density_plot

# Making 2D plot functions for different observable pairs
mt_density_plot = obs_2d_density_plot_maker([momentum, temperature], ['$<X_n>$', 'Temperature'])
me_density_plot = obs_2d_density_plot_maker([momentum, energy], ['$<X_n>$', '$\mathcal{E}$'])
et_density_plot = obs_2d_density_plot_maker([energy, temperature], ['$\mathcal{E}$', 'Temperature'])

def obs_2d_density_from_list_plot_maker(observables, names=None): # Similar to above but works for lists rather than datasets with a 'realisation dimension'
    """
    Makes function that will plot density of two particular observables from a list of datasets.
    observables, list - Two observables functions e.g. [energy, temperature]
    names, list - Two strings to label axis.
    """
    def obs_2d_density_plot(ds_list, fax=None, nbins=50, x_lims=None, y_lims=None, color_min=0, colorbar=True, *args, **kwargs):
        "Takes a list of xr.ds and plots 2D density of observable points"

        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        # Get data points
        xs = []
        ys = []
        for ds in ds_list:
            x = observables[0](ds).values.flatten()
            y = observables[1](ds).values.flatten()
            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        # Use KDE density
        fig, ax = kde_density_2d_plot(xs, ys, fax=fax, nbins=nbins, x_lims=x_lims, y_lims=y_lims, color_min=color_min,
                        colorbar=colorbar, shading='auto', *args, **kwargs)
        if names is not None:
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
        return fig, ax
    return obs_2d_density_plot

# Making 2D plot functions for different observable pairs
ET_density_from_list_plot = obs_2d_density_from_list_plot_maker([energy, temperature], ['$\mathcal{E}$', 'Temperature'])
MT_density_from_list_plot = obs_2d_density_from_list_plot_maker([momentum, temperature], ['$<X_n>$', 'Temperature'])
EM_density_from_list_plot = obs_2d_density_from_list_plot_maker([energy, momentum], ['$\mathcal{E}$', '$<X_n>$'])

###########################################
### 3D Probabilistic Plots
###########################################

def EMT_sliced_density_plot(instanton_ds, eps=0.25, di_max=None, T_slices=None, fax=None, cmap = plt.cm.PuRd, **kwargs):

    # init Axis
    if fax is None:
        fig, ax = init_3d_fax()
    else:
        fig, ax = fax

    # Plotting Density Slices

    # Loop 1: Figure out denisty range so that colors in different slices mean the same thing
    di_max_list = []
    for T0 in T_slices:
        # Get Energy and Momentum data in range [T - eps, T + eps]
        reduced_data = instanton_ds.where(np.logical_and(instanton_ds.T>=T0-eps, instanton_ds.T<=T0+eps), drop=True)
        m = momentum(reduced_data).values
        m = m[~np.isnan(m)]
        e = energy(reduced_data).values
        e = e[~np.isnan(e)]

        # Evaluate 2D KDE Density on a grid
        density = kde_density_2d(m, e)
        m_bounds = [m.min(), m.max()]
        e_bounds = [e.min(), e.max()]
        nbins=100
        mi, ei = np.mgrid[m_bounds[0]:m_bounds[1]:nbins*1j, e_bounds[0]:e_bounds[1]:nbins*1j]
        di = density(np.vstack([mi.flatten(), ei.flatten()]))
        di_max_list += [di.max()]

    if di_max is None:
        di_max = max(di_max_list)

    # Loop 2: Plot the Slices
    for i, T0 in enumerate(T_slices[T_slices.argsort()]):
        # Get Energy and Momentum data in range [T - eps, T + eps]
        reduced_data = instanton_ds.where(np.logical_and(instanton_ds.T>=T0-eps, instanton_ds.T<=T0+eps), drop=True)
        m = momentum(reduced_data).values
        m = m[~np.isnan(m)]
        e = energy(reduced_data).values
        e = e[~np.isnan(e)]

        m_bounds = [m.min(), m.max()]
        e_bounds = [e.min(), e.max()]

        # Evaluate 2D KDE Density on a grid
        density = kde_density_2d(m, e)
        nbins=100
        mi, ei = np.mgrid[m_bounds[0]:m_bounds[1]:nbins*1j, e_bounds[0]:e_bounds[1]:nbins*1j]
        di = density(np.vstack([mi.flatten(), ei.flatten()]))
        di = di.reshape(mi.shape)

        # Filter Colors to only show density core
        cmap = plt.cm.PuRd
        norm = cm.Normalize(vmin=0.2, vmax=di_max)
        colors = cmap(norm(di))
        colors[:,...,-1] = np.where(di>0.2, 0.5, 0)

        # 3d Plot ISO Surface
        ax.plot_surface(mi, ei, np.full_like(mi, T0), facecolors=colors, shade=False, zorder=i+1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))


    return fig, ax

###########################################
### Probabilistic Section Plots
###########################################

def EM_section_plots(instanton_ds, T_slices, eps=0.25, di_max=None, m_bounds=None, e_bounds=None, show_scatter=False):
    no_of_slices = len(T_slices)
    fig, axes = plt.subplots(2, int(no_of_slices/2), figsize = (25, 10))

    # Loop 1: Figure out denisty range so that colors in different slices mean the same thing
    di_max_list = []
    m_bounds_list = []
    e_bounds_list = []
    for T0 in T_slices:
        # Get Energy and Momentum data in range [T - eps, T + eps]
        reduced_data = instanton_ds.where(np.logical_and(instanton_ds.T>=T0-eps, instanton_ds.T<=T0+eps), drop=True)
        m = momentum(reduced_data).values
        m = m[~np.isnan(m)]
        e = energy(reduced_data).values
        e = e[~np.isnan(e)]

        # Evaluate 2D KDE Density on a grid
        density = kde_density_2d(m, e)
        if m_bounds is None:
            m_bounds = [m.min(), m.max()]
            m_bounds_list += m_bounds
        if e_bounds is None:
            e_bounds = [e.min(), e.max()]
            e_bounds_list += e_bounds
        nbins=100
        mi, ei = np.mgrid[m_bounds[0]:m_bounds[1]:nbins*1j, e_bounds[0]:e_bounds[1]:nbins*1j]
        if di_max is None:
            di = density(np.vstack([mi.flatten(), ei.flatten()]))
            di_max_list += [di.max()]

    if m_bounds is None:
        print('a')
        m_bounds = [min(m_bounds_list), max(m_bounds_list)]
    if e_bounds is None:
        e_bounds = [min(e_bounds_list), max(e_bounds_list)]
    if di_max is None:
        di_max = max(di_max_list)

    # Loop 2: Doing the Plots
    for i, T0 in enumerate(T_slices):
        ax = axes.flatten()[i]
        # Get Energy and Momentum data in range [T - eps, T + eps]
        reduced_data = instanton_ds.where(np.logical_and(instanton_ds.T>=T0-eps, instanton_ds.T<=T0+eps), drop=True)
        m = momentum(reduced_data).values
        m = m[~np.isnan(m)]
        e = energy(reduced_data).values
        e = e[~np.isnan(e)]

        # Evaluate 2D KDE Density on a grid
        density = kde_density_2d(m, e)

        nbins=100
        mi, ei = np.mgrid[m_bounds[0]:m_bounds[1]:nbins*1j, e_bounds[0]:e_bounds[1]:nbins*1j]
        di = density(np.vstack([mi.flatten(), ei.flatten()]))
        di = di.reshape(mi.shape)
        ax.pcolor(mi, ei, di, cmap='PuRd', vmax = di_max, shading='auto')
        if show_scatter:
            ax.scatter(m, e, c='b', alpha=0.1, facecolor=None)
        ax.set_title(f'T = {T0:.3g}')
        ax.set_ylabel('$\\mathcal{E}$')
        ax.set_xlabel('$\\mathcal{M}$')
        ax.set_xlim(m_bounds)
        ax.set_ylim(e_bounds)


    cmap = plt.cm.PuRd
    norm = cm.Normalize(vmin=0., vmax=di_max)
    fig.suptitle('Mean-Energy Density Slices', fontsize=20)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label='$\\rho$')
    return fig, axes

def MT_local_density_plot(ds, T_slices=None, fax=None, **kwargs):#m_lims=None, T_lims=None):

    # init axis
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    if T_slices is None:
        T_slices = np.linspace(ds.T.min().item(), ds.T.max().item(), 10)

    # Loop to plot densities normalised to individual slices
    for i in range(len(T_slices)-1):

        # Pick data within a bounded region
        lower_T = T_slices[i]
        upper_T = T_slices[i + 1]
        data_in_slice = ds.where(np.logical_and(ds.T>=lower_T, ds.T<=upper_T), drop=True)
        m = momentum(data_in_slice).values
        m = m[~np.isnan(m)]
        if len(m) != 0:
            T = temperature(data_in_slice).values
            T = T[~np.isnan(T)]
            kde_density_2d_plot(m, T, shading='auto', fax=fax, colorbar=None, **kwargs)# x_lims=m_lims, y_lims=T_lims, color_min=0.005, ax=ax, shading='auto', colorbar=False, cmap='PuRd')
    return fig, ax

def ET_local_density_plot(ds, T_slices=None, fax=None, **kwargs):

    # init axis
    if fax is None:
        fax = init_2d_fax()
    else:
        fig, ax = fax

    if T_slices is None:
        T_slices = np.linspace(ds.T.min().item(), ds.T.max().item(), 10)

    # Loop to plot densities normalised to individual slices
    for i in range(len(T_slices)-1):

        # Pick data within a bounded region
        lower_T = T_slices[i]
        upper_T = T_slices[i + 1]
        data_in_slice = ds.where(np.logical_and(ds.T>=lower_T, ds.T<=upper_T), drop=True)
        e = energy(data_in_slice).values
        e = e[~np.isnan(e)]
        if len(e) != 0:
            T = temperature(data_in_slice).values
            T = T[~np.isnan(T)]
            kde_density_2d_plot(e, T, shading='auto', fax=fax, colorbar=None, **kwargs)
    return fig, ax

def get_data_within_T_slice(ds, lower_T, upper_T):
    "Gets all data witing "
    if (ds.T.min() <= upper_T) & (ds.T.max() >= lower_T):
        return ds.where(np.logical_and(ds.T>=lower_T, ds.T<=upper_T), drop=True)
    else:
        return None

def get_observable_within_T_slice(obs, ds, lower_T, upper_T):
    "Gets observable data points within a given range from a ds."
    data_within_slice = get_data_within_T_slice(ds, lower_T, upper_T)
    if data_within_slice is not None:
        return obs(data_within_slice)

def get_observable_witihin_T_slice_from_list(obs, ds_list, lower_T, upper_T):
    "Loops through a ds list and gets observable data points within a temperature range."
    total_data_in_slice = []
    for ds in ds_list:
        ds_data_in_slice = get_observable_within_T_slice(obs, ds, lower_T, upper_T)
        if ds_data_in_slice is not None:
            total_data_in_slice.append(ds_data_in_slice)
    return np.concatenate(total_data_in_slice)


def ET_local_density_from_list_plot(ds_list, no_of_slices=None, T_slices=None, fax=None, **kwargs):
    "Plots density in ET projection from a list of paths."

    # Init axis
    if fax is None:
        fax = init_2d_fax()
        fig, ax = fax
    else:
        fig, ax = fax

    # T-Section initialisation
    T_min = 1000
    E_min = 100
    T_max = 0
    E_max = 0
    for ds in ds_list:
        T_min = min(T_min, ds.T.min().item())
        E_min = min(E_min, energy(ds).min().item())
        T_max = max(T_max, ds.T.max().item())
        E_max = max(E_max, energy(ds).max().item())

    if T_slices is None:
        if no_of_slices is not None:
            T_slices = np.linspace(T_min, T_max, no_of_slices)
        elif no_of_slices is None:
            T_slices = np.linspace(T_min, T_max, 10)

    # Loop through slices getting data in each slice
    for i in range(len(T_slices)-1):
        lower_T = T_slices[i]
        upper_T = T_slices[i + 1]
        E = get_observable_witihin_T_slice_from_list(energy, ds_list, lower_T, upper_T)
        T = get_observable_witihin_T_slice_from_list(temperature, ds_list, lower_T, upper_T)
        # h = ax.hist2d(E, T, **kwargs)
        kde_density_2d_plot(E, T, shading='auto', fax=fax, colorbar=None, **kwargs)

    ax.set_ylim(T_min, T_max)
    ax.set_xlim(E_min, E_max)
    ax.set_ylabel('$\\mathcal{T}$')
    ax.set_xlabel('$\\mathcal{E}$')
    return

def MT_local_density_from_list_plot(ds_list, no_of_slices=None, T_slices=None, fax=None, **kwargs):
    "Plots density in MT projection from a list of paths."

    # Init axis
    if fax is None:
        fax = init_2d_fax()
        fig, ax = fax
    else:
        fig, ax = fax

    # T-Section initialisation
    T_min = 1000
    M_min = 100
    T_max = 0
    M_max = 0
    for ds in ds_list:
        T_min = min(T_min, ds.T.min().item())
        M_min = min(M_min, momentum(ds).min().item())
        T_max = max(T_max, ds.T.max().item())
        M_max = max(M_max, momentum(ds).max().item())

    if T_slices is None:
        if no_of_slices is not None:
            T_slices = np.linspace(T_min, T_max, no_of_slices)
        elif no_of_slices is None:
            T_slices = np.linspace(T_min, T_max, 10)

    # Loop through slices getting data in each slice
    for i in range(len(T_slices)-1):
        lower_T = T_slices[i]
        upper_T = T_slices[i + 1]
        M = get_observable_witihin_T_slice_from_list(momentum, ds_list, lower_T, upper_T)
        T = get_observable_witihin_T_slice_from_list(temperature, ds_list, lower_T, upper_T)
        kde_density_2d_plot(M, T, shading='auto', fax=fax, colorbar=None, **kwargs)

    ax.set_ylim(T_min, T_max)
    ax.set_xlim(M_min, M_max)
    ax.set_ylabel('$\\mathcal{T}$')
    ax.set_xlabel('$\\mathcal{M}$')
    return
