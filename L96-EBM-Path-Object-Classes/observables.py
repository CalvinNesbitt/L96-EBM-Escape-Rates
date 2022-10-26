"""
Observables of the L96 EBM System that we will look at
"""

def momentum(ds):
    "ds is instanton"
    return ds.X.mean(dim='space')

def energy(ds):
    return 0.5 * (ds.X**2).mean(dim='space')

def temperature(ds):
    return ds.T

def observe_list(observable, ds_list):
    observations = []
    for ds in ds_list:
        observations.append(observable(ds))
    return observations    
