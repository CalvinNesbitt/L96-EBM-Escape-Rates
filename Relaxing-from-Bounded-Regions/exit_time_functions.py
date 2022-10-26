"""
Function to determine when we have left the transient.
"""
import numpy as np

def ds_as_np(ds):
    "Return the data as numpy array."
    X = ds.X.values
    T = ds.T.values
    return np.column_stack((X, T))

def check_w_exit_time(transient_ds, lb=272.28):
    transient =  ds_as_np(transient_ds)
    exit_index =  transient.shape[0] - np.argmax(np.flip(transient.T[-1] > lb))
    return transient_ds.time[exit_index].item()

def check_sb_exit_time(transient_ds, ub=267.66):
    transient =  ds_as_np(transient_ds)
    exit_index =  transient.shape[0] - np.argmax(np.flip(transient.T[-1] < ub))
    return transient_ds.time[exit_index].item()
