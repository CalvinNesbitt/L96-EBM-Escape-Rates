"""
Functions used to compute escape rates.
"""

from personal_stats import linear_regression_plot, linear_regression_fit
import numpy as np

def surviving_trajectories(x, n):
    """
    How many of the escape times x are larger than n.
    x, np.array: Array of escape times.
    n, float: The length of time within the M-state
    """
    return np.sum(x>n)

def survival_counts(x, ns):
    "See surviving_trajectories function. Computes for many n values."
    return np.array([surviving_trajectories(x, n) for n in ns])

def n_star(x, ns):
    """
    The first time a trajectory leaves the M-state.
    x, np.array: Array of escape times.
    ns, np.array: Array of possible escape times that we count the survival count at.
    """
    scs = survival_counts(x, ns) # How many survive at each n
    return ns[scs < len(x)][0] # when is scs < len(x) for first time, i.e a trajectory has left

def escape_rate(x, ns=None, min_sc=0, max_sc=None):
    if max_sc is None:
        max_sc = len(x) # max survival count used for the fit
    if ns is None:
        ns = np.linspace(0, x.max(), 100) # default times we count survival count at
    scs = survival_counts(x, ns)
    indexes_for_fit = (scs < max_sc) * (scs > min_sc) # Only fit line when scs in specified range
    line, lr = linear_regression_fit(ns[indexes_for_fit], np.log(scs[indexes_for_fit]))
    return -lr.slope, lr

def estimated_mean_occupation_time(x):
    ns = np.linspace(0, x.max(), 100)
    scs = survival_counts(x, ns)
    indexes_for_fit = (scs < len(x)) * (scs > k)
    line, lr = linear_regression_fit(ns[indexes_for_fit], np.log(scs[indexes_for_fit]))
    escape_rate = -lr.slope
    return 1/escape_rate + n_star(x, ns)

def escape_rate_plot(x, fax=None, ns=None, min_sc=0, max_sc=None, rate_label=None, **kwargs):
    fig, ax = fax
    if max_sc is None:
        max_sc = len(x) # max survival count used for the fit
    if ns is None:
        ns = np.linspace(0, x.max(), 100) # default times we count survival count at
    scs = survival_counts(x, ns)
    indexes_for_fit = (scs < max_sc) * (scs > min_sc) # Only fit line when scs in specified range
    line, lr = linear_regression_fit(ns[indexes_for_fit], np.log(scs[indexes_for_fit]))

    if rate_label is None:
        rate_label = f'$\\kappa=${-lr.slope:.2f}'

    ax.plot(ns[indexes_for_fit], line(ns[indexes_for_fit]), 'r--', alpha=1.0, lw=3, label=rate_label)
    ax.scatter( ns, np.nan_to_num(np.log(scs), neginf=0), **kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel('log(Survival Count)')
    ax.legend()
    return fax
