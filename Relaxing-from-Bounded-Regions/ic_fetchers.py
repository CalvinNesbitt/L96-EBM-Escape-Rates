"""
Functions for randomly sampling from bounded region where we expect transient to be.
"""

########################################################################
## Imports
########################################################################
import numpy as np
import numpy.random as rm
import os
from input_output import *

########################################################################
## IC fetchers
########################################################################
S_w_critical = 7.8 # Roughly where the W-Attractor Dissapears
S_sb_critical = 15.2 # Roughly where the SB-Attractor Dissapears

def get_attractor_bounds(attractor, S):
    "Find box bounding attractor for a particular value of S."
    # Fetch attractor
    if attractor == 'w':
        attractor = get_w_attractor(S=S)
    elif attractor == 'sb':
        attractor = get_sb_attractor(S=S)
    elif attractor == 'm':
        attractor = get_m_state(S=S)

    # Compute lower/upper bounds for X and T variables
    lower_bounds = np.min(ds_to_np(attractor), axis=0)
    upper_bounds = np.max(ds_to_np(attractor), axis=0)
    return lower_bounds, upper_bounds

def get_sample_ic_from_bounding_box(attractor, S, number_of_samples):
    "Fetches samples from a box."
    lb, ub = get_attractor_bounds(attractor, S)
    return rm.uniform(lb, ub, size=(number_of_samples, 51))

def make_sb_edge_sample_ic(number_of_samples):
    return get_sample_ic_from_bounding_box('sb', S_sb_critical, number_of_samples)

def make_w_edge_sample_ic(number_of_samples):
    return get_sample_ic_from_bounding_box('w', S_w_critical, number_of_samples)

def load_sb_edge_ic():
    return np.load(ic_save_dir  + 'sb-edge-ic.npy')

def load_w_edge_ic():
    return np.load(ic_save_dir  + 'w-edge-ic.npy')


if __name__ == '__main__':
    "Save ic as np files"
    sb_edge_ic = make_sb_edge_sample_ic(1000)
    w_edge_ic  = make_w_edge_sample_ic(1000)

    if not os.path.exists(ic_save_dir):
        os.makedirs(ic_save_dir)

    np.save(ic_save_dir  + 'sb-edge-ic', sb_edge_ic)
    np.save(ic_save_dir  + 'w-edge-ic', w_edge_ic)
