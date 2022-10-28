"""
Classes for computing transient lifetime escape rates and how they scale with S
"""

##########################################
## Imports
##########################################

# Standard
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import linregress
import glob

from plotting import *
from escapeRateCalculations import *

##########################################
## Functions for fetching data
##########################################
escape_time_data_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Escape-Rates/Transient-Lifetime-Data/'

def get_SB_escape_time_data(pd=escape_time_data_pd):
    return transientLifetimeCollection(glob.glob(escape_time_data_pd + '*/SB*/*.txt'))

def get_W_escape_time_data(pd=escape_time_data_pd):
    return transientLifetimeCollection(glob.glob(escape_time_data_pd + '*/W*/*.txt'))

##########################################
## Information used for scaling fit
##########################################

S_w_critical = 7.8 # Roughly where the W-Attractor Dissapears
S_sb_critical = 15.2 # Roughly where the SB-Attractor Dissapears

def power_law(x, a, b):
    return a*np.power(x, b)


##########################################
## Transient Lifetime Classes
##########################################

class transientLifetimes(pd.core.series.Series):
    """
    Class takes transient lifetime data and computes escape rate & produces basic plots.
    """

    def __init__(self, data_file):
        super().__init__(np.loadtxt(data_file, skiprows=1), name='Transient-Lifetimes')

        # Unpack info in Name
        S_str, self.disappearing_attractor = data_file.split('S_')[1].split('-Transient')[0].split('/')
        S = float(S_str.replace('_', '.'))
        self.attrs['S'] = S
        self.S = S
        self.name = 'Transient Lifetimes'

        # Min/Max Survival Counts used in Escape Rate Computation
        self.min_sc = 0
        self.max_sc = None
        self.survival_count_times = np.linspace(0, self.max(), 100)

    def histogram_plot(self, *args, grid_points=100, fax=None, **kwargs):
        "Uses kde to plot histogram of transient lifetimes"

        # Init fax
        if fax is None:
            fax = init_2d_fax()
        fig, ax = fax

        # Plot Histogram
        kde = stats.gaussian_kde(self.values)
        xs = np.linspace(0, self.max() + self.std(), grid_points)
        ax.plot(xs, kde(xs), *args, **kwargs)
        ax.set_xlabel(self.name)
        ax.set_ylabel('$\\rho$')
        ax.set_title(f'S = {self.S:.3f}, Transient Lifetime Histogram')
        return fig, ax

    @property
    def escape_rate(self):
        return escape_rate(self.values, ns=self.survival_count_times,
                           min_sc=self.min_sc, max_sc=self.max_sc)[0]

    @property
    def escape_rate_info(self):
        return escape_rate(self.values, ns=self.survival_count_times,
                           min_sc=self.min_sc, max_sc=self.max_sc)[1]

    def escape_rate_fit_plot(self, fax=None, **kwargs):
        # Init fax
        if fax is None:
            fax = init_2d_fax()
        fig, ax = fax
        ax.set_title(f'S={self.S:.3f}, Escape Rate Fit')

        return escape_rate_plot(self.values, fax=fax, ns=self.survival_count_times,
                           min_sc=self.min_sc, max_sc=self.max_sc, rate_label=None, **kwargs)


class transientLifetimeCollection:
    """
    Class takes collection of transient lifetime data and computes how escape rates scale with S.
    """

    def __init__(self, file_list):
        self._transient_lifetimes = [transientLifetimes(file) for file in file_list]
        self.disappearing_attractor = self._transient_lifetimes[0].disappearing_attractor
        if self.disappearing_attractor == 'SB':
            self.S_crit = S_sb_critical
        elif self.disappearing_attractor == 'W':
            self.S_crit = S_w_critical

    def __getitem__(self, idx):
        return self._transient_lifetimes[idx]

    def __len__(self):
        return len(self._transient_lifetimes)

    @property
    def S_values(self):
        return np.array([x.S for x in self._transient_lifetimes])

    @property
    def distances_from_S_crit(self):
        return np.abs(self.S_crit - self.S_values)

    @property
    def escape_rates(self):
        return np.array([x.escape_rate for x in self._transient_lifetimes])

    @property
    def mean_lifetimes(self):
        return np.array([x.mean() for x in self._transient_lifetimes])

    @property
    def critical_exponent(self):
        pars, cov = curve_fit(power_law, self.distances_from_S_crit, self.escape_rates)
        return pars[1]

    def S_vs_mean_lifetime_plot(self, *args, fax=None, **kwargs):

        # Init fax
        if fax is None:
            fax = init_2d_fax()
        fig, ax = fax

        ax.scatter(self.S_values, self.mean_lifetimes, *args, **kwargs)
        ax.set_xlabel('S')
        ax.set_ylabel('$<\\tau>$')
        ax.set_title('Mean Lifetime as a Function of S')
        return fax

    def S_vs_escape_rate_plot(self, *args, fax=None, **kwargs):

        # Init fax
        if fax is None:
            fax = init_2d_fax()
        fig, ax = fax

        ax.scatter(self.S_values, self.escape_rates, *args, **kwargs)
        ax.set_xlabel('S')
        ax.set_ylabel('$\\kappa$')
        ax.set_title('Escape Rate as a Function of S')
        return fax

    def S_distance_vs_escape_rate_plot(self, *args, fax=None, **kwargs):

        # Init fax
        if fax is None:
            fax = init_2d_fax()
        fig, ax = fax

        ax.scatter(self.distances_from_S_crit, self.escape_rates, *args, **kwargs)
        ax.set_xlabel('$|S - S_{crit}|$')
        ax.set_ylabel('$\\kappa$')
        ax.set_title('Escape Rate as a Function of S')
        return fax

    def scaling_law_fit_plot(self, *args, critical_exponent_label=None, fax=None, **kwargs):

        # Init fax
        if fax is None:
            fax = init_2d_fax()
        fig, ax = fax

        # Perform fit
        pars, cov = curve_fit(power_law, self.distances_from_S_crit, self.escape_rates)

        if critical_exponent_label is None:
            critical_exponent_label = f'$\\gamma=${pars[1]:.2f}'

        # Plot Line of best Fit
        xs = np.linspace(self.distances_from_S_crit.min(), self.distances_from_S_crit.max(), 50)
        ax.plot(xs, power_law(xs, pars[0], pars[1]), 'r--',
                alpha=1.0, lw=2, label=critical_exponent_label)

        # Plot Escape Rate Data
        ax.scatter(self.distances_from_S_crit, self.escape_rates, *args, **kwargs)
        ax.set_xlabel('$|S - S_{crit}|$')
        ax.set_ylabel('$\\kappa$')
        ax.set_title(f'Critical Exponent fit')
        ax.legend()
        return fax
