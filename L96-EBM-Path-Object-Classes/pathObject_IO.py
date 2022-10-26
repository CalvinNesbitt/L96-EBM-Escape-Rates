"""
Contains:
- Locations of attractor and relaxation data.
- Functions for helping us to open attractor and relaxation data.
"""
########################################################################
## Imports
########################################################################
# Standard package imports
import xarray as xr
import numpy as np

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
    # Location of Relaxation Data
    relxation_data_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Relaxations/Relaxation-Data/'
    relaxation_from_mstate_pd = relxation_data_pd + '/Relaxing-from-M-State/'
    relaxation_from_cube_pd = relxation_data_pd + '/Relaxing-from-Cube/'
    # Location of Attractor Data: Should be SB, W and M.
    attractor_data_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton/Deterministic-Model/Effect-of-S/Attractor-Data/'
else:
    # Location of Relaxation Data
    # Location of Attractor Data: Should be SB, W and M.
    attractor_data_pd = '/rds/general/user/cfn18/home/Instantons/L96-EBM-Effect-of-S/Attractor-Data/'

# Attractor File Names
def w_attractor_file_name(pd=attractor_data_pd, S=10):
    return pd + f'/S_{S}/hot-attractor/L96-EBM-Trajectory1.nc'

def sb_attractor_file_name(pd=attractor_data_pd, S=10):
    return pd + f'/S_{S}/cold-attractor/L96-EBM-Trajectory1.nc'

def m_state_file_name(pd=attractor_data_pd, S=10):
    return pd + f'/S_{S}/M-State/1.nc'

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
