"""
Common tasks when doing stats in python
"""

# Standard Package imports
import numpy as np
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt


def kde_density_2d(x, y):
    " Returns KDE density funciton from a 2D dataset of the form (xi, yi)."
    xy = np.vstack([x,y])
    density = stats.gaussian_kde(xy)
    return density

def linear_regression_fit(X, Y):
    """
    Fits Y = M * X + C using scipy.stats linregress:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

    Inputs
    --------------
    X, np.array - X data points.
    Y, np.array - Y data points.

    Returns
    --------------
    line_of_best_fit, function - Function that evaluates line of best fit.
    lr_result, LinregressResult - Scipy linregress results object. Contains R and p values.
    """
    lr_result = linregress(X, Y)
    def line_of_best_fit(x):
        return lr_result.slope * x + lr_result.intercept
    return [line_of_best_fit, lr_result]

def linear_regression_plot(X, Y, x_range=None, fax=None, param_values=True):
    """
    Plots line of best fit.

    Inputs
    --------------
    X, np.array - X data points.
    Y, np.array - Y data points.
    x_range, np.array - points where we evaluate line fo best fit.
    param_values, boolean = include M/c values?
    """

    # Initialise Axis
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    if x_range is None:
        x_range = np.linspace(np.min(X), np.max(X))

    lr_fit = linear_regression_fit(X, Y)
    linear_model = lr_fit[0]
    lr_object = lr_fit[1]

    ax.plot(x_range, linear_model(x_range), 'r--')
    ax.scatter(X, Y)
    ax.grid()

    if param_values:

        textstr = f'$R^2$ = {lr_object.rvalue **2:.2f}\np = {lr_object.pvalue:.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    return fig, ax

def lagged_auto_correlation(x, lag=0):
    "Computes lagged autocovariance of a timeseries with itself"
    if lag == 0:
        return np.corrcoef(x)
    else:
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]

def lagged_auto_correlations(x, dt=None, lags=[0]):
    """
    Computes lagged autocovariance for many lags.
    - If user specifies dt it returns corresponding time points.
    - Assumption is constant time spacing.
    """

    # Computing lagged auto correlations
    lagged_acs = []
    for lag in lags:
        lac = lagged_auto_correlation(x, lag=lag)
        lagged_acs.append(lac)

    # Outputting time if user specifies dt
    if dt != None:
        time_points = np.arange(0, dt * (len(lags)), dt)
        return time_points, np.array(lagged_acs)
    else:
        return np.array(lagged_acs)

def integrated_autocorrelation(x, dt, lags):
    "Computes integrated autocorrelation assuming constant dt."
    lacs = lagged_auto_correlations(x, lags=lags)
    return np.trapz(lacs, dx=dt)

def autocorrelation_plot(x, lags, dt, fax=None, iac=True, **kwargs):

    # Initialise Axis
    if fax is None:
        fig, ax = init_2d_fax()
    else:
        fig, ax = fax

    # Get lagged auto correlations
    time, lacs = lagged_auto_correlations(x, dt=dt, lags=lags)

    # Plotting
    ax.plot(time, lacs, **kwargs)
    ax.set_xlabel('Time')
    ax.grid()

    if iac:
        textstr = f'Integrated autocorrleation:{integrated_autocorrelation(x, dt, lags):.3f}'
        ax.set_title(textstr)
    return fig, ax
