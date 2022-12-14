"""
I/O for scripts generating transient lifetime data.
"""
########################################################################
## Imports
########################################################################
# Standard package imports
import xarray as xr
import numpy as np
import sys
import os

# Function to identify where we're running
from pathlib import Path
mac_home_dir = Path('/Users/cfn18')
home_dir = Path.home()
def on_mac():
    return mac_home_dir == home_dir

########################################################################
## Data + Custom import Locations
########################################################################

# Data parent adn attractor object directories
if on_mac():
    # PD for Attractor Data: Should be SB, W and M.
    attractor_data_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Effect-of-S/Attractor-Data/'
    attractorObjectDirectory = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Escape-Rates/L96-EBM-Path-Object-Classes/'
    escape_time_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Escape-Rates/Transient-Lifetime-Data/'
    ic_save_dir = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Escape-Rates/Relaxing-from-Bounded-Regions/IC-Files/'
else:
    attractor_data_pd = '/rds/general/user/cfn18/home/Instantons/L96-EBM-Effect-of-S/Attractor-Data/'
    attractorObjectDirectory = '/rds/general/user/cfn18/home/Instantons/L96-EBM-Escape-Rates/L96-EBM-Path-Object-Classes/'
    escape_time_pd = '/rds/general/user/cfn18/home/Instantons/L96-EBM-Escape-Rates/Transient-Lifetime-Data/'
    ic_save_dir = '/rds/general/user/cfn18/home/Instantons/L96-EBM-Escape-Rates//Relaxing-from-Bounded-Regions/IC-Files/'


########################################################################
## Escape Time File Names
########################################################################
def escape_time_directory(S, attractor_transient, pd=escape_time_pd):
    return pd + f'/S_{S:.3f}/{attractor_transient.upper()}-Transient/'.replace('.', '_')

def transient_path_folder(*args, **kwargs):
    return escape_time_directory(*args, **kwargs) + '/Transient-Paths/'

def escape_time_file(*args, **kwargs):
    return escape_time_directory(*args, **kwargs) + '/Transient-Lifetimes.txt'

def initialise_escape_time_file(*args, **kwargs):
    sd = escape_time_directory(*args, **kwargs)
    if not os.path.exists(sd):
        os.makedirs(sd, exist_ok = True)
    file_name = escape_time_file(*args, **kwargs)
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('Transient-Lifetimes\n')
            f.close()
    return

def save_transient_lifetime(lifetime, *args, **kwargs):
    file_name = escape_time_file(*args, **kwargs)
    with open(file_name, 'a') as f:
        f.write(str(lifetime) + '\n')
        f.close()
    return

########################################################################
## Attractor File Names
########################################################################

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

def ds_to_np(ds):
    "Converts ds point to np array."
    X = ds.X.values
    T = ds.T.values
    return np.column_stack((X, T))
