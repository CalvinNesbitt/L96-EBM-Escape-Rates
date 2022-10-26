"""
Contains:
- Locations of attractor and relaxation data.
- Functions for helping us to open attractor and relaxation data.
"""
########################################################################
## Standard Imports
########################################################################
# Standard package imports
import xarray as xr
import numpy as np
import sys
import glob

# Function to identify where we're running
from pathlib import Path
mac_home_dir = Path('/Users/cfn18')
home_dir = Path.home()
def on_mac():
    return mac_home_dir == home_dir

########################################################################
## Data Locations
########################################################################

# Data parent directory
if on_mac():
    # PD for Attractor Data: Should be SB, W and M.
    attractor_data_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Effect-of-S/Attractor-Data/'

else:
    print('Please specify location of attractor data.')

# Attractor File Names
def attractor_folder_name(end_state, pd=attractor_data_pd, S=10):
    "end_state, string: One of w, sb or m."
    return pd + f'/S_{S:.3f}/{end_state.upper()}-State/'.replace('.', '_')

def w_attractor_file_name(pd=attractor_data_pd, S=10):
    sd = attractor_folder_name('w', pd=pd, S=S)
    return sd + '1.nc'

def sb_attractor_file_name(pd=attractor_data_pd, S=10):
    sd = attractor_folder_name('sb', pd=pd, S=S)
    return sd + '1.nc'

def m_state_file_name(pd=attractor_data_pd, S=10):
    sd = attractor_folder_name('m', pd=pd, S=S)
    return sd + '1.nc'

########################################################################
## Custom Imports
########################################################################
sys.path.append(path_and_plots_files)
from plots import *

########################################################################
## Functions for opening relaxations
########################################################################
sys.path.append(path_and_plots_files)
from relaxationPaths import *

def get_relaxations(S, end_state, pd=relaxation_from_mstate_pd):
    file_list = glob.glob(relaxation_sub_dir(S, end_state, pd=pd) + '*.nc')
    return L96EBM_RelaxationPathCollection(file_list, direction=f'm2{end_state.lower()}')

def get_all_relaxations(S, pd=relaxation_from_mstate_pd):
    "Function for getting relaxations in both directions"
    relaxations = get_relaxations(S, '*', pd=pd)

    # Update with required info that isn't present when you do mixed direction load
    for x in relaxations._path_list:
        if x.ds.T.values[-1] > x.ds.T.values[0]:
            x.direction = 'm2w'
            x.end_state = 'w'
        else:
            x.direction = 'm2sb'
            x.end_state = 'sb'
        x.exit_index = x.m_exit_index()
    return relaxations


########################################################################
## Functions for opening determinsitic attractors & M-State
########################################################################

def get_attractor(file_name):
    return xr.open_dataset(file_name)

def get_w_attractor(pd=attractor_data_pd, S=10):
    file_name = w_attractor_file_name(pd=pd, S=S)
    return get_attractor(file_name)

def get_sb_attractor(pd=attractor_data_pd, S=10):
    file_name = sb_attractor_file_name(pd=pd, S=S)
    return get_attractor(file_name)

def get_m_state(pd=attractor_data_pd, S=10):
    file_name = m_state_file_name(pd=pd, S=S)
    return get_attractor(file_name)

def get_smooth_m_state(pd=attractor_data_pd, S=10):
    file_name = m_state_file_name(pd=pd, S=S)
    m_state = get_attractor(file_name)
    return m_state.interp({'time':np.arange(5.1, 105, 0.01)}, method='cubic')

def ds_to_np(ds):
    "Converts ds point to np array."
    X = ds.X.values
    T = ds.T.values
    return np.append(X, T)

def get_ds_points(ds, n):
    "Sample n points from a ds without replacament."
    time_points = np.random.choice(ds.time, n, replace=False)
    return [ds_to_np(ds.sel(time=x)) for x in time_points]

########################################################################
## Functions for saving relaxations
########################################################################

def relaxation_sub_dir(S, end_state, pd=relxation_data_pd):
    return pd + f'/S_{S:.3f}/End-State_{end_state}/'.replace('.', '_')
