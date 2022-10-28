"""
Functions for fetching escape rate data.
"""

##########################################
## File Locations
##########################################

escape_time_data_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Escape-Rates/Transient-Lifetime-Data/'
transient_class_objects_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Escape-Rates/Transient-Lifetime-Classes/'

##########################################
## Imports
##########################################
# Standard
import sys
import glob

# Custom Code
sys.path.append(transient_class_objects_pd)
from transientLifetimes import *

##########################################
## Functions for fetching data
##########################################

def get_SB_transient_lifetime_data(pd=escape_time_data_pd):
    return transientLifetimeCollection(glob.glob(escape_time_data_pd + '*/SB*/*.txt'))

def get_W_transient_lifetime_data(pd=escape_time_data_pd):
    return transientLifetimeCollection(glob.glob(escape_time_data_pd + '*/W*/*.txt'))
