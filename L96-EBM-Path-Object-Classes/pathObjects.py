"""
Base Classes for paths in the L96-EBM.

Contents:

- L96EBMPath, Path in the L96 EBM model.
    Base class that others are built on.
    Has functionality to plot, clip & compute T-conditional path.
"""

##########################################
## Imports
##########################################
# Standard Package Imports
import xarray as xr
import sys
import numpy as np

# Custom Imports
from plots import *

def points_within_T_slice(ds, temp_thresholds):
    ds.load()
    return ds.where(np.logical_and(ds.T>=temp_thresholds[0], ds.T<=temp_thresholds[1]), drop=True)

##########################################
## L96 EBM Path Class Definition
##########################################

class L96EBMPath:
    """
    Path in the L96 EBM model.
    """

    def __init__(self, file_location, direction=None):
        """
        Parameters
        -----------
        file_location, string
            Location of the .netcdf file containing L96-EBM path.
        direction, string
            Which asymptotic states the path connects.
            Form is 'a2b' e.g. 'sb2m' or 'sb2w'.
            If not specifies attempts to determine from T values.
        """
        # Unpack path information
        self.file_location = file_location
        self.ds = xr.open_dataset(file_location)
        self.attrs = self.ds.attrs
        self.S = self.attrs['S']

        # Determine Start/End States
        if direction is None:
            self._determine_direction()
        else:
            self.direction = direction
        self.start_state = self.direction.split('2')[0]
        self.end_state = self.direction.split('2')[1]

    def _determine_direction(self):
        "Determine direction if user doesn't specify."
        if self.ds.T.values[0] < self.ds.T.values[-1]:
            self.direction = 'sb2w'
        if self.ds.T.values[0] > self.ds.T.values[-1]:
            self.direction = 'w2sb'
        return

    def clip_path(self, i, j, normalise_time = True):
        "Reduce path to just points those between time indices i and j."
        dt = (self.ds.time[1] - self.ds.time[0]).item() # assuming constant dt
        self.ds = self.ds.isel(time=slice(i, j))
        if normalise_time:
            self.ds = self.ds.assign_coords({'time': dt * np.arange(len(self.ds.time))})
        return

    @property
    def ds_as_np(self):
        "Return the data as numpy array."
        X = self.ds.X.values
        T = self.ds.T.values
        return np.column_stack((X, T))

    def get_observable_points_as_np(self, obs):
        return obs(self.ds).values.flatten()

    def get_observable_points_in_T_slice(self, obs, temp_thresholds):
        return obs(points_within_T_slice(self.ds, temp_thresholds))

    def _plot_relevant_atttractors(self, plot, fax, attractors=None):
        if attractors is None:
            attractors = [self.start_state, self.end_state]
        attractor_list = []
        color_list = []
        # Determine attractors to be plotted
        if ('sb' in attractors):
            attractor_list.append(get_sb_attractor(S=self.S))
            color_list.append('b')
        if ('m' in attractors):
            attractor_list.append(get_smooth_m_state(S=self.S))
            color_list.append('g')
        if ('w' in attractors):
            attractor_list.append(get_w_attractor(S=self.S))
            color_list.append('r')
        # Plot attractors
        for attractor, color in zip(attractor_list, color_list):
            plot(attractor, c=color, fax=fax)
        return

    def _2d_plot_projection(self, plot, fax=None, attractors=None, *args, **kwargs):
        "2D plot of data in a certain projection"
        if fax is None:
            fig, ax = init_2d_fax()
            fax = [fig, ax]
        else:
            fig, ax = fax
        plot(self.ds, fax=fax, *args, **kwargs)
        self._plot_relevant_atttractors(plot, fax, attractors=attractors)
        ax.grid()
        return fig, ax

    def ET_plot(self, fax=None, *args, **kwargs):
        fig, ax = self._2d_plot_projection(ET_plot, fax=fax, *args, **kwargs)
        return fig, ax

    def MT_plot(self, *args, fax=None, **kwargs):
        self._2d_plot_projection(MT_plot, fax=fax, *args, **kwargs)
        return

    def EM_plot(self, fax=None, *args, **kwargs):
        self._2d_plot_projection(EM_plot, fax=fax, *args, **kwargs)
        return

    def T_timeseries_plot(self, fax=None, *args, **kwargs):
        T_timeseries_plot(self.ds, fax=fax, *args, **kwargs)
        return

    def E_timeseries_plot(self, fax=None, *args, **kwargs):
        E_timeseries_plot(self.ds, fax=fax, *args, **kwargs)
        return

    def M_timeseries_plot(self, fax=None, *args, **kwargs):
        M_timeseries_plot(self.ds, fax=fax, *args, **kwargs)
        return

    def X_hvm_plot(self, *args, fax=None, **kwargs):

        # Initialise fax
        if fax == None:
            fax = init_2d_fax()
        fig, ax = fax

        # Reinterpolate data so plot looks better
        interpolated_ds = self.ds.interp(time=np.arange(min(self.time), max(self.time), 0.1), space=np.arange(1, 50, 0.1))
        interpolated_ds.X.plot(*args, ax=ax, **kwargs)
        return fig, ax

    def plot_point_projection(self, plot, time_index, *args, fax=None, attractors=['m'], marker='.', ms=10, **kwargs):
        "2D plot of IC in a certain projection"
        if fax is None:
            fig, ax = init_2d_fax()
            fax = [fig, ax]
        else:
            fig, ax = fax
        self._plot_relevant_atttractors(plot, fax, attractors=attractors)
        point = self.ds.isel(time=time_index)
        plot(point, *args, fax=fax, marker=marker, ms=ms, ls='', **kwargs)
        ax.grid()
        return fig, ax

    def plot_point_MT(self, time_index, *args, fax=None, attractors=['m'], marker='.', ms=10, **kwargs):
        fig, ax = self.plot_point_projection(MT_plot, time_index, *args, fax=fax, attractors=attractors, marker=marker, ms=ms, **kwargs)
        return fig, ax

    def plot_point_ET(self, time_index, *args, fax=None, attractors=['m'], marker='.', ms=10, **kwargs):
        fig, ax = self.plot_point_projection(ET_plot, time_index,  *args, fax=fax, attractors=attractors, marker=marker, ms=ms, **kwargs)
        return fig, ax

    def plot_ic_MT(self, *args, **kwargs):
        return self.plot_point_MT(0, *args, **kwargs)

    def plot_ic_ET(self, *args, **kwargs):
        return self.plot_point_ET(0, *args, **kwargs)

class L96EBM_PathCollection:
    """
    List of paths in the L96 EBM model. Contains a variety of plotting and clipping functionality.
    """

    def __init__(self, file_locations, direction=None):
        self.file_locations = file_locations
        self._path_list = self._get_paths_init(direction)
        self.direction = self._path_list[0].direction
        self.start_state = self._path_list[0].start_state
        self.end_state = self._path_list[0].end_state
        self.attrs = self._path_list[0].ds.attrs
        self.S = self.attrs['S']

    def _get_paths_init(self, direction):
        path_list = []
        for file in self.file_locations:
            try:
                path_list.append(L96EBMPath(file, direction=direction))
            except:
                print(f'No file at {file}.\n')
        return path_list

    def __getitem__(self, idx):
        return self._path_list[idx]

    def __len__(self):
        return len(self._path_list)

    def clip_paths(self, i_list, j_list, normalise_time = True):

        # Case where user provides int indices for indexing
        if type(i_list) is int:
            i_list = [i_list for path in self._path_list]
        if type(j_list) is int:
            j_list = [j_list for path in self._path_list]

        # Clipping paths in loop
        for n, path in enumerate(self._path_list):
            i = i_list[n]
            j = j_list[n]
            path.clip_path(i, j, normalise_time=normalise_time)
        return

    def get_observable_points_as_np(self, obs):
        all_points = []
        for path in self._path_list:
            path_points = path.get_observable_points_as_np(obs)
            all_points.append(path_points)
        return np.concatenate(all_points)

    def get_observable_points_in_T_slice(self, obs, temp_thresholds):
        all_points = []
        for path in self._path_list:
            path_points = path.get_observable_points_in_T_slice(obs, temp_thresholds).values.flatten()
            all_points.append(path_points)
        return np.concatenate(all_points)

    ## Defining a function to get the T conditional path
    def calculate_T_conditional_path(self, T_slices=10, min_samples=10, pretty_start=False, pretty_end=False):
        self.T_conditional_path  = TemperatureConditionalPath(self, T_slices=T_slices, min_samples=min_samples, pretty_start=pretty_start, pretty_end=pretty_end)

    @property
    def ds_list(self):
        return [path.ds for path in self._path_list]

    def ET_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        for path in self._path_list:
            path.ET_plot(fax=[fig, ax], *args, **kwargs)
        return

    def MT_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        for path in self._path_list:
            path.MT_plot(fax=[fig, ax], *args, **kwargs)
        return

    def EM_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        for path in self._path_list:
            path.EM_plot(fax=[fig, ax], *args, **kwargs)
        return

    def T_timeseries_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        for path in self._path_list:
            path.T_timeseries_plot(fax=[fig, ax], *args, **kwargs)
        return

    def E_timeseries_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        for path in self._path_list:
            path.E_timeseries_plot(fax=[fig, ax], *args, **kwargs)
        return

    def M_timeseries_plot(self, fax=None, *args, **kwargs):
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        for path in self._path_list:
            path.M_timeseries_plot(fax=[fig, ax], *args, **kwargs)
        return

    def ET_density_plot(self, fax=None, *args, **kwargs):
        ET_density_from_list_plot(self.ds_list, fax=fax, *args, **kwargs)
        return

    def ET_sliced_density_plot(self, no_of_slices=None, T_slices=None, fax=None, **kwargs):
        ET_local_density_from_list_plot(self.ds_list, no_of_slices=no_of_slices,
                                        T_slices=T_slices, fax=fax, **kwargs)
        return

    def MT_sliced_density_plot(self, no_of_slices=None, T_slices=None, fax=None, **kwargs):
        MT_local_density_from_list_plot(self.ds_list, no_of_slices=no_of_slices,
                                    T_slices=T_slices, fax=fax, **kwargs)
        return


class TemperatureConditionalPath:
    "Path generated from conditioning on temperature slices in L96 EBM path"

    def __init__(self, path_collection, T_slices=10, min_samples=10, pretty_start=False, pretty_end=False):
        self.path_collection = path_collection
        self.attrs = self.path_collection.attrs
        self.direction = self.path_collection.direction
        self.S = self.path_collection.S
        self.start_state = self.path_collection.start_state
        self.end_state = self.path_collection.end_state
        self._calulate_T_conditional_path(self.path_collection, T_slices, min_samples, pretty_start=pretty_start, pretty_end=pretty_end)

    def recalculate(self, T_slices=10, min_samples=10, pretty_start=False, pretty_end=False):
        self.T_slices = T_slices
        self._calulate_T_conditional_path(self.path_collection, T_slices = T_slices, min_samples = min_samples, pretty_start=pretty_start, pretty_end=pretty_end)
        return

    @property
    def pretty_end_point(self):
        # Determine End Point
        attractors= {'w':get_w_attractor(S=self.S),
        'sb':get_sb_attractor(S=self.S),
        'm':get_m_state(S=self.S)}
        attractor = attractors[self.end_state]
        return np.array([energy(attractor).mean(), momentum(attractor).mean(), temperature(attractor).mean()])

    @property
    def pretty_start_point(self):
        # Determine Start Point
        attractors= {'w':get_w_attractor(S=self.S),
        'sb':get_sb_attractor(S=self.S),
        'm':get_m_state(S=self.S)}
        attractor = attractors[self.start_state]
        return np.array([energy(attractor).mean(), momentum(attractor).mean(), temperature(attractor).mean()])

    def _calulate_T_conditional_path(self, path_collection, T_slices=10, min_samples=10, pretty_start=False, pretty_end=False):

        # Initialising data we want to keep
        conditional_energy_points = []
        conditional_momentum_points = []
        conditional_temperature_points = []

        conditional_energy_means = []
        conditional_momentum_means = []
        conditional_temperature_means = []

        conditional_energy_spread = []
        conditional_momentum_spread = []
        conditional_temperature_spread = []

        # Default T slices, if int provided we do evenly spaced partition
        if type(T_slices) == int:
            attractors= {'w':get_w_attractor(S=self.S),
            'sb':get_sb_attractor(S=self.S),
            'm':get_m_state(S=self.S)}
            start_attractor = attractors[self.start_state]
            end_attractor = attractors[self.end_state]
            temp_min = min(temperature(start_attractor).mean(),  temperature(end_attractor).mean())
            temp_max = max(temperature(start_attractor).mean(),  temperature(end_attractor).mean())
            T_slices = np.linspace(temp_min, temp_max, T_slices)

        # Looping through temp slices and plotting conditional means
        for i in range(len(T_slices) - 1):
            # Getting observables within in temperature slice
            lower_T = T_slices[i]
            upper_T = T_slices[i + 1]
            energy_in_T_slice = path_collection.get_observable_points_in_T_slice(energy, [lower_T, upper_T])
            momentum_in_T_slice = path_collection.get_observable_points_in_T_slice(momentum, [lower_T, upper_T])
            temperature_in_T_slice = path_collection.get_observable_points_in_T_slice(temperature, [lower_T, upper_T])

            # Plotting Mean within T slice using ax.errorbar
            if len(temperature_in_T_slice) > min_samples:
                conditional_energy_points.append(energy_in_T_slice)
                conditional_momentum_points.append(momentum_in_T_slice)
                conditional_temperature_points.append(temperature_in_T_slice)

                conditional_energy_means.append(energy_in_T_slice.mean())
                conditional_momentum_means.append(momentum_in_T_slice.mean())
                conditional_temperature_means.append(temperature_in_T_slice.mean())

                conditional_energy_spread.append(energy_in_T_slice.std())
                conditional_momentum_spread.append(momentum_in_T_slice.std())
                conditional_temperature_spread.append(temperature_in_T_slice.std())

        # Updating Calculated Info
        self.T_slices = T_slices
        self.energy_points = conditional_energy_points
        self.momentum_points = conditional_momentum_points
        self.temperature_points = conditional_temperature_points
        self.path =  np.stack((conditional_energy_means, conditional_momentum_means, conditional_temperature_means), axis=1)
        self.path_spread = np.stack((conditional_energy_spread, conditional_momentum_spread, conditional_temperature_spread), axis=1)

        # Adding pretty start/end point
        # Determining where to insert Start/End point According to T-Value
        T_difference = np.abs(self.pretty_end_point - self.path).T[-1]
        if T_difference[0] > T_difference[-1]: # Fancy end point should be at end of path
            start_point = self.pretty_start_point
            start_spread = self.path_spread[0]
            end_point = self.pretty_end_point
            end_spread = self.path_spread[-1]
        else:
            start_point = self.pretty_end_point
            start_spread = self.path_spread[-1]
            end_point = self.pretty_start_point
            end_spread = self.path_spread[0]

        if pretty_end:
            self.path = np.vstack((self.path, end_point))
            self.path_spread = np.vstack((self.path_spread, end_spread))
        if pretty_start:
            self.path = np.vstack((start_point, self.path))
            self.path_spread = np.vstack((start_spread, self.path_spread))
        return

    def MT_pipe_plot(self, fax=None, attractors=None):
        # Setting up figure
        if fax is None:
            fig, ax = init_2d_fax()
            fax=[fig, ax]
        else:
            fig, ax = fax
        # Plot attractors
        self.path_collection[0]._plot_relevant_atttractors(MT_plot, fax=fax, attractors=attractors)
        # Set Default Color
        color_dictionary = {'sb': 'b', 'w':'r', 'm':'g'}
        c = color_dictionary[self.start_state]
        self.custom_MT_pipe_plot(fax=fax, color = c, alpha=0.2, line_kwargs={'c':c, 'lw':3, 'ls':'--'})
        # Set Arrow Direction
        direction_dictionary = {
        'sb2m': 'up',
        'sb2w': 'up',
        'm2sb': 'down',
        'm2w': 'up',
        'w2m': 'down',
        'w2sb': 'down'}
        direction = direction_dictionary[self.direction]
        if direction == 'down':
            si = int(self.path.shape[0]/2)
            ei = si - 1
        if direction == 'up':
            si = int(self.path.shape[0]/2)
            ei = si + 1
        self.add_MT_arrows(fax=fax, color=c, head_length=1, head_width=0.3, length_includes_head=True, start_index=si, end_index=ei)
        return

    def custom_MT_pipe_plot(self, fax=None, mean_line=True, line_kwargs={},  **kwargs):

        # Setting up figure
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        ax.grid()
        ax.set_xlabel('$\mathcal{M}$')
        ax.set_ylabel('$T$')

        # Plotting Conditional Means

        M = self.path.T[1]
        M_std  = self.path_spread.T[1]
        T = self.path.T[-1]
        ax.fill_betweenx(T, M - M_std, M + M_std, **kwargs)

        if mean_line:
            ax.plot(M, T, **line_kwargs)
        return

    def add_MT_arrows(self, fax, start_index=None, end_index=None, **kwargs):
        fig, ax = fax
        M = self.path.T[1]
        T = self.path.T[-1]
        if start_index is None:
            mid_point = int(len(T)/2)
            if self.direction == 'c2h':
                start_index = mid_point
                end_index = mid_point + 1
            elif self.direction == 'h2c':
                start_index = mid_point - 1
                end_index = mid_point

        M_start = M[start_index ]
        M_next = M[end_index]
        T_start = T[start_index ]
        T_next = T[end_index]
        ax.arrow(M_start, T_start, M_next - M_start, T_next - T_start, **kwargs)
        return

    def add_ET_arrows(self, fax, start_index=None, end_index=None, **kwargs):
        fig, ax = fax
        E = self.path.T[0]
        T = self.path.T[-1]
        if start_index is None:
            mid_point = int(len(T)/2)
            if self.direction == 'c2h':
                start_index = mid_point
                end_index = mid_point + 1
            elif self.direction == 'h2c':
                start_index = mid_point - 1
                end_index = mid_point
        E_start = E[start_index]
        E_next = E[end_index]
        T_start = T[start_index]
        T_next = T[end_index]
        ax.arrow(E_start, T_start, E_next - E_start, T_next - T_start, **kwargs)
        return


    def MT_scatter_plot(self, fax=None, means=True, scatter=True, error_bars=True, **kwargs):

        # Setting up figure
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        ax.grid()
        ax.set_xlabel('$\mathcal{M}$')
        ax.set_ylabel('$T$')

        # Plotting Conditional Means scatter
        for M, T in zip(self.momentum_points, self.temperature_points):

            if scatter: # plot all data
                ax.scatter(M, T, c='k', alpha=0.5)
            if error_bars:
                ax.errorbar(M.mean(), T.mean(), yerr=T.std(), xerr=M.std(), **kwargs)
            else:
                ax.scatter(M.mean(), T.mean(), **kwargs)
        return

    def ET_pipe_plot(self, fax=None, attractors=None):
        # Setting up figure
        if fax is None:
            fig, ax = init_2d_fax()
            fax=[fig, ax]
        else:
            fig, ax = fax

        # Plot attractors
        self.path_collection[0]._plot_relevant_atttractors(ET_plot, fax=fax, attractors=attractors)

        # Set Default Color
        color_dictionary = {'sb': 'b', 'w':'r', 'm':'g'}
        c = color_dictionary[self.start_state]
        self.custom_ET_pipe_plot(fax=fax, color = c, alpha=0.2, line_kwargs={'c':c, 'lw':3, 'ls':'--'})
        # Set Arrow Direction
        direction_dictionary = {
        'sb2m': 'up',
        'sb2w': 'up',
        'm2sb': 'down',
        'm2w': 'up',
        'w2m': 'down',
        'w2sb': 'down'}
        direction = direction_dictionary[self.direction]
        if direction == 'down':
            si = int(self.path.shape[0]/2)
            ei = si - 1
        if direction == 'up':
            si = int(self.path.shape[0]/2)
            ei = si + 1
        self.add_ET_arrows(fax=fax, color=c, head_length=1, head_width=0.8, length_includes_head=True, start_index=si, end_index=ei)
        return

    def custom_ET_pipe_plot(self, fax=None, mean_line=True, line_kwargs={},  **kwargs):

        # Setting up figure
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        ax.grid()
        ax.set_xlabel('$\mathcal{E}$')
        ax.set_ylabel('$T$')

        # Plotting Conditional Means
        E = self.path.T[0]
        E_std  = self.path_spread.T[0]
        T = self.path.T[-1]
        ax.fill_betweenx(T, E - E_std, E + E_std, **kwargs)

        if mean_line:
            ax.plot(E, T, **line_kwargs)
        return


    def ET_scatter_plot(self, fax=None, means=True, scatter=True, error_bars=True, **kwargs):

        # Setting up figure
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax
        ax.grid()
        ax.set_xlabel('$\mathcal{E}$')
        ax.set_ylabel('$T$')

        # Plotting Conditional Eeans scatter
        for E, T in zip(self.energy_points, self.temperature_points):

            if scatter: # plot all data
                ax.scatter(E, T, c='k', alpha=0.5)
            if error_bars:
                ax.errorbar(E.mean(), T.mean(), yerr=T.std(), xerr=E.std(), **kwargs)
            else:
                ax.scatter(E.mean(), T.mean(), **kwargs)

        return

    def EMT_plot(self, fax=None, **kwargs):

        # Setting up figure
        if fax is None:
            fig, ax = init_3d_fax()
        else:
            fig, ax = fax
        # Plotting Path
        E = self.path.T[0]
        M = self.path.T[1]
        T = self.path.T[-1]
        ax.plot(M, E, T, **kwargs)
        return

    def plot_T_slices(self, extent=[0, 20], fax=None, **kwargs):

        # Setting up figure
        if fax is None:
            fig, ax = init_2d_fax()
        else:
            fig, ax = fax

        for T in self.T_slices:
            x = np.linspace(extent[0], extent[1], 50)
            y = np.full_like(x, T)
            ax.plot(x, y, **kwargs)
        return
