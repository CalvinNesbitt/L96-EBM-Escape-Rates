"""
Functions to determine when we went enter/exit a certain state.
"""
##########################################
## Imports
##########################################

# Standard Package Imports
import sys
import numpy as np

# Custom code imports
from observables import *
from pathObject_IO import get_m_state, get_sb_attractor, get_w_attractor

###########################################
## Functions used to define entry/exit to rectangles around attractors
##########################################

def observable_thresholds(obs, cold, S=10, extension=0.5):
    """
    obs - what observable we want the bounds of.
    cold - True/False/'M-State', which attractor or M-state we are bounding.
    S - solar parameter
    extension - uniform addition to bounds
    """

    if cold == 'M-State':
        attractor = get_m_state(S=S)
        return [obs(attractor).min().item() - extension, obs(attractor).max().item() + extension]
    if cold:
        attractor = get_sb_attractor(S=S)
    else:
        attractor = get_w_attractor(S=S)
    return [obs(attractor).min().item() - extension, obs(attractor).max().item() + extension]

def temperature_thresholds(cold, S=10, extension=0.1):
    return observable_thresholds(temperature, cold=cold, S=S, extension=extension)

def mean_thresholds(cold, S=10, extension=0.1):
    return observable_thresholds(momentum, cold=cold, S=S, extension=extension)

def energy_thresholds(cold, S=10, extension=0.1):
    return observable_thresholds(energy, cold=cold, S=S, extension=extension)

def sb_exit_index(ds, E_extension=0., M_extension=0., T_extension=0.):
    S = int(ds.S)
    cold=True

    # Points within the temperature bounds
    temp_thresholds = temperature_thresholds(cold=cold, S=S, extension=T_extension)
    points_above_lower_threshold = ds.T.values > temp_thresholds[0]
    points_below_upper_threshold = ds.T.values < temp_thresholds[1]
    points_within_temp_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the M bounds
    m_thresholds = mean_thresholds(cold=cold, S=S, extension=M_extension)
    points_above_lower_threshold = momentum(ds).values > m_thresholds[0]
    points_below_upper_threshold = momentum(ds).values < m_thresholds[1]
    points_within_mean_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the E bounds
    e_thresholds = energy_thresholds(cold=cold, S=S, extension=E_extension)
    points_above_lower_threshold = energy(ds).values > e_thresholds[0]
    points_below_upper_threshold = energy(ds).values < e_thresholds[1]
    points_within_energy_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Return last time you leave EMT box
    points_in_cold_EMT_box = points_within_temp_bounds * points_within_mean_bounds * points_within_energy_bounds
    return np.max(np.nonzero(points_in_cold_EMT_box))

def sb_entry_index(ds, E_extension=0., M_extension=0., T_extension=0.):
    S = int(ds.S)
    cold=True

    # Points within the temperature bounds
    temp_thresholds = temperature_thresholds(cold=cold, S=S, extension=T_extension)
    points_above_lower_threshold = ds.T.values > temp_thresholds[0]
    points_below_upper_threshold = ds.T.values < temp_thresholds[1]
    points_within_temp_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the M bounds
    m_thresholds = mean_thresholds(cold=cold, S=S, extension=M_extension)
    points_above_lower_threshold = momentum(ds).values > m_thresholds[0]
    points_below_upper_threshold = momentum(ds).values < m_thresholds[1]
    points_within_mean_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the E bounds
    e_thresholds = energy_thresholds(cold=cold, S=S, extension=E_extension)
    points_above_lower_threshold = energy(ds).values > e_thresholds[0]
    points_below_upper_threshold = energy(ds).values < e_thresholds[1]
    points_within_energy_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Return last time you leave EMT box
    points_in_cold_EMT_box = points_within_temp_bounds * points_within_mean_bounds * points_within_energy_bounds
    return np.min(np.nonzero(points_in_cold_EMT_box))

def w_exit_index(ds, E_extension=0., M_extension=0., T_extension=0.):
    S = int(ds.S)
    cold=False

    # Points within the temperature bounds
    temp_thresholds = temperature_thresholds(cold=cold, S=S, extension=T_extension)
    points_above_lower_threshold = ds.T.values > temp_thresholds[0]
    points_below_upper_threshold = ds.T.values < temp_thresholds[1]
    points_within_temp_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the M bounds
    m_thresholds = mean_thresholds(cold=cold, S=S, extension=M_extension)
    points_above_lower_threshold = momentum(ds).values > m_thresholds[0]
    points_below_upper_threshold = momentum(ds).values < m_thresholds[1]
    points_within_mean_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the E bounds
    e_thresholds = energy_thresholds(cold=cold, S=S, extension=E_extension)
    points_above_lower_threshold = energy(ds).values > e_thresholds[0]
    points_below_upper_threshold = energy(ds).values < e_thresholds[1]
    points_within_energy_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Return last time you leave EMT box
    points_in_cold_EMT_box = points_within_temp_bounds * points_within_mean_bounds * points_within_energy_bounds
    return np.max(np.nonzero(points_in_cold_EMT_box))

def w_entry_index(ds, E_extension=0., M_extension=0., T_extension=0.):
    S = int(ds.S)
    cold=False

    # Points within the temperature bounds
    temp_thresholds = temperature_thresholds(cold=cold, S=S, extension=T_extension)
    points_above_lower_threshold = ds.T.values > temp_thresholds[0]
    points_below_upper_threshold = ds.T.values < temp_thresholds[1]
    points_within_temp_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the M bounds
    m_thresholds = mean_thresholds(cold=cold, S=S, extension=M_extension)
    points_above_lower_threshold = momentum(ds).values > m_thresholds[0]
    points_below_upper_threshold = momentum(ds).values < m_thresholds[1]
    points_within_mean_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the E bounds
    e_thresholds = energy_thresholds(cold=cold, S=S, extension=E_extension)
    points_above_lower_threshold = energy(ds).values > e_thresholds[0]
    points_below_upper_threshold = energy(ds).values < e_thresholds[1]
    points_within_energy_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Return last time you leave EMT box
    points_in_cold_EMT_box = points_within_temp_bounds * points_within_mean_bounds * points_within_energy_bounds
    return np.min(np.nonzero(points_in_cold_EMT_box))

def M_entry_index(ds, T_extension=0.):
    "Works solely based on time you enter temperature threhsold"
    S = int(ds.S)
    cold='M-State'

    # Points within the temperature bounds
    temp_thresholds = temperature_thresholds(cold=cold, S=S, extension=T_extension)
    if ds.T.values[0] < temp_thresholds[0]: #i.e. we start cold
        points_above_lower_threshold = ds.T.values > temp_thresholds[1]
        return np.min(np.nonzero(points_above_lower_threshold))

    elif ds.T.values[0] > temp_thresholds[1]: #i.e. we start hot
        points_below_upper_threshold = ds.T.values < temp_thresholds[0]
        return np.min(np.nonzero(points_below_upper_threshold))

# Functions that get observable bounds of stochastic cold/hot attractor

def stochastic_observable_thresholds(obs, cold, S=10, eps = 0.5, extension=0.5):
    """
    obs - what observable we want the bounds of.
    eps - which Invairant measure we open
    cold - True/False/'M-State', which attractor or M-state we are bounding.
    S - solar parameter
    extension - uniform addition to bounds
    """
    attractor = Invariant_Measure(S=S, cold=cold, eps=eps).data
    return [obs(attractor).values.min().item() - extension, obs(attractor).values.max() + extension]


def stochastic_temperature_thresholds(cold, S=10, eps=0.5, extension=0.1):
    return stochastic_observable_thresholds(temperature, cold=cold, S=S,eps=eps, extension=extension)

def stochastic_mean_thresholds(cold, S=10, eps=0.5, extension=0.1):
    return stochastic_observable_thresholds(momentum, cold=cold, S=S,eps=eps, extension=extension)

def stochastic_energy_thresholds(cold, S=10, eps=0.5, extension=0.1):
    return stochastic_observable_thresholds(energy, cold=cold, S=S, eps=eps, extension=extension)

def stochastic_sb_exit_time(ds, eps=0.5, E_extension=0., M_extension=0., T_extension=0.):
    S = int(ds.S)
    cold=True

    # Points within the temperature bounds
    temp_thresholds = stochastic_temperature_thresholds(cold=cold, S=S, eps=eps, extension=T_extension)
    points_above_lower_threshold = ds.T.values > temp_thresholds[0]
    points_below_upper_threshold = ds.T.values < temp_thresholds[1]
    points_within_temp_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the M bounds
    m_thresholds = stochastic_mean_thresholds(cold=cold, S=S, eps=eps, extension=M_extension)
    points_above_lower_threshold = momentum(ds).values > m_thresholds[0]
    points_below_upper_threshold = momentum(ds).values < m_thresholds[1]
    points_within_mean_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Points within the E bounds
    e_thresholds = stochastic_energy_thresholds(cold=cold, S=S, eps=eps, extension=E_extension)
    points_above_lower_threshold = energy(ds).values > e_thresholds[0]
    points_below_upper_threshold = energy(ds).values < e_thresholds[1]
    points_within_energy_bounds = points_below_upper_threshold * points_above_lower_threshold

    # Return last time you leave EMT box
    points_in_cold_EMT_box = points_within_temp_bounds * points_within_mean_bounds * points_within_energy_bounds
    return np.max(np.nonzero(points_in_cold_EMT_box))
