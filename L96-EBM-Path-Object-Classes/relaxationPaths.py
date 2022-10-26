"""
Contains classes for fetching relaxations. These build on the l96EBMPathObject classes.
"""

########################################################################
## Standard Imports
########################################################################
import sys
import glob
import numpy as np

########################################################################
## Custom Code Imports
########################################################################
from pathObjects import L96EBMPath, L96EBM_PathCollection
from entry_exit_functions import sb_entry_index, w_entry_index, temperature_thresholds
from plots import kde_plot, init_2d_fax
from observables import momentum, energy
from escapeRateCalculations import n_star, escape_rate, escape_rate_plot

########################################################################
## Relaxation Path Definitions
########################################################################

class L96EBM_RelaxationPath(L96EBMPath):

    def __init__(self, file_location, direction=None):
        super().__init__(file_location, direction=direction)

        # Determine Entry/Exit indexes
        self.exit_index = self.m_exit_index()
        if self.end_state == 'w':
            self.entry_index = w_entry_index(self.ds) + 1
        if self.end_state == 'sb':
            self.entry_index = sb_entry_index(self.ds) + 1

    def m_exit_index(self, **kwargs):
        temp_thresholds = temperature_thresholds('M-State', S=self.S, extension=0)
        if self.end_state == 'w':
            points_above_upper_threshold = self.ds.T.values > max(temp_thresholds)
            return np.argmax(points_above_upper_threshold)
        if self.end_state == 'sb':
            points_below_lower_threshold = self.ds.T.values < min(temp_thresholds)
            return np.argmax(points_below_lower_threshold)

    def recompute_entry_index(self, **kwargs):
        if self.end_state == 'w':
            self.entry_index = w_entry_index(self.ds, **kwargs) + 1
            return
        if self.end_state == 'sb':
            self.entry_index = sb_entry_index(self.ds, **kwargs) + 1
            return

    def clip_m_to_sb(self):
        # First clip to last exit time
        exit_index = self.exit_index
        self.clip_path(self.exit_index, -1)
        # Then find first entry time
        self.clip_path(0, self.entry_index)
        return

    def clip_m_to_w(self):
        # First clip to last exit time
        exit_index = self.exit_index
        self.clip_path(self.exit_index, -1)
        # Then find first entry time
        self.clip_path(0, self.entry_index)
        return

    @property
    def time(self):
        return self.ds.time

    @property
    def relaxation_time(self):
        return self.time[self.entry_index].item()

    @property
    def exit_time(self):
        return self.time[self.exit_index].item()

    @property
    def m_state_occupation_time(self):
        return self.time[self.exit_index].item()
    @property
    def exit_point(self):
        point = self.ds.isel(time=self.exit_index)
        return point

    @property
    def entry_point(self):
        point = self.ds.isel(time=self.entry_index)
        return point

    def plot_exit_point_MT(self, *args, attractors=None, **kwargs):
        if attractors == None:
            attractors = ['m']
        return self.plot_point_MT(self.exit_index, *args, attractors=attractors, **kwargs)

    def plot_exit_point_ET(self, *args, attractors=None, **kwargs):
        return self.plot_point_ET(self.exit_index, *args, **kwargs)

    def plot_entry_point_MT(self, *args, attractors=None, **kwargs):
        if attractors == None:
            attractors = [self.end_state]
        return self.plot_point_MT(self.entry_index, *args, attractors=attractors, **kwargs)

    def plot_entry_point_ET(self, *args, **kwargs):
        return self.plot_point_ET(self.entry_index, *args, **kwargs)

class L96EBM_RelaxationPathCollection(L96EBM_PathCollection):

    def __init__(self, file_locations, direction=None):
        # Don't Call Super as we want list members to relaxation type
        self.file_locations = file_locations
        self._path_list = self._get_paths_init(direction)
        self.direction = self._path_list[0].direction
        self.start_state = self._path_list[0].start_state
        self.end_state = self._path_list[0].end_state
        self.attrs = self._path_list[0].ds.attrs
        self.S = self.attrs['S']
        self.min_sc = 0
        self.max_sc = None

    def _get_paths_init(self, direction):
        path_list = []
        for file in self.file_locations:
            try:
                path_list.append(L96EBM_RelaxationPath(file, direction=direction))
            except:
                print(f'No file at {file}.\n')
        return path_list

    def clip_m_to_w(self):
        for i, path in enumerate(self._path_list):
            try:
                path.clip_m_to_w()
            except:
                print(f'Path {i} could not be clipped.')

    def clip_m_to_sb(self):
        for i, path in enumerate(self._path_list):
            try:
                path.clip_m_to_sb()
            except:
                print(f'Path {i} could not be clipped.')

    @property
    def relaxation_times(self):
        return np.array([x.relaxation_time for x in self._path_list])

    def relaxtion_times_density_plot(self, *args, grid_points=50, fax=None, **kwargs):
        x_lbl = f' {self.direction.upper()} relaxation times'
        return kde_plot(self.relaxation_times, *args, grid_points=grid_points, fax=fax, x_label=x_lbl, **kwargs)

    @property
    def m_state_occupation_times(self):
        return np.array([x.m_state_occupation_time for x in self._path_list])

    def m_state_occupation_times_density_plot(self, *args, grid_points=50, fax=None, **kwargs):
        x_lbl = f'{self.direction.upper()} M-State occupation times'
        return kde_plot(self.m_state_occupation_times, *args, grid_points=grid_points, fax=fax, x_label=x_lbl, **kwargs)

    def surviving_trajectories(self, n):
        "How many trajectories are still in the m-state after time n."
        return np.sum(self.m_state_occupation_times > n)

    def survival_counts(self, ns):
        return np.array([surviving_trajectories(self.m_state_occupation_times, n) for n in ns])

    @property
    def exit_points(self):
        return [x.exit_point for x in self._path_list]

    @property
    def entry_points(self):
        return [x.entry_point for x in self._path_list]

    def _exit_point_density(self, observable, *args, **kwargs):
        exit_data = np.array([observable(x).item() for x in self.exit_points])
        return kde_plot(exit_data, *args, **kwargs)

    def exit_point_density_M(self, *args, **kwargs):
        return self._exit_point_density(momentum, *args, **kwargs)

    def exit_point_density_E(self, *args, **kwargs):
        return self._exit_point_density(energy, *args, **kwargs)

    def _entry_point_density(self, observable, *args, **kwargs):
        entry_data = np.array([observable(x).item() for x in self.entry_points])
        return kde_plot(entry_data, *args, **kwargs)

    def entry_point_density_M(self, *args, **kwargs):
        return self._entry_point_density(momentum, *args, **kwargs)

    def entry_point_density_E(self, *args, **kwargs):
        return self._entry_point_density(energy, *args, **kwargs)

    def escape_rate_plot(self, fax=None, ns=None, **kwargs):
        if fax is None:
            fax = init_2d_fax()
        fig, ax = fax
        return escape_rate_plot(self.m_state_occupation_times, fax=fax, ns=ns, min_sc=self.min_sc, max_sc=self.max_sc, **kwargs)

    def escape_rate(self, ns=None):
        return escape_rate(self.m_state_occupation_times, ns=ns, min_sc=self.min_sc, max_sc=self.max_sc)[0]

    def escape_rate_info(self, ns=None):
        return escape_rate(self.m_state_occupation_times, ns=ns, min_sc=self.min_sc, max_sc=self.max_sc)[1]
