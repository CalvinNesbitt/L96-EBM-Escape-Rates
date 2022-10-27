"""
In this script we run an ensemble of relaxations with ic that are randomly sampled points within a specified bounded region. These relaxations will be used to determine the escape rates.
"""
########################################################################
## Imports
########################################################################
# Standard Packages
import numpy.random as rm
import itertools

# Custom Code Imports
from l96EBM import *
from ic_fetchers import *
from input_output import *
from exit_time_functions import *

########################################################################
## Parameter Setup
########################################################################
input_number = int(sys.argv[1]) - 1

# W Setups
w_S_values = np.linspace(7.5, 7.7, 10)
w_setups = list( itertools.product(w_S_values, ['w'], np.arange(0, 100) ) )

# SB Setups
sb_S_values = np.linspace(15.3, 15.5, 10)
sb_setups = list( itertools.product(sb_S_values, ['sb'], np.arange(0, 100) ) )

all_setups = w_setups + sb_setups
setup = all_setups[input_number]
print(len(all_setups))
S, disapearing_attractor, ic_number = setup

# Transient Length
max_transient_length = 10000
dt = 0.01 # Time between observations
save_transients = False
number_of_observations = int(max_transient_length/dt)

########################################################################
## Where we save data
########################################################################
save_file = escape_time_file(S, disapearing_attractor)
if not os.path.exists(save_file):
    initialise_escape_time_file(S, disapearing_attractor)

########################################################################
## Load IC File,  Exit time function & Exit Time File
########################################################################
if disapearing_attractor == 'w':
    ic_list = load_w_edge_ic()
    exit_function = check_w_exit_time
elif disapearing_attractor == 'sb':
    ic_list = load_sb_edge_ic()
    exit_function = check_sb_exit_time

ic = ic_list[ic_number]
#########################################################################
## Running and Saving Transient Escape Time
#########################################################################

# Setup Integrator
runner = Integrator()
runner.S = S
runner.set_state(ic)
looker = TrajectoryObserver(runner)

print(f'**Running Integration with S = {S:.3f}.\nIC number is {ic_number}.\nInvestigating {disapearing_attractor} transient.\nLoaded ic from {ic_save_dir}.\n\n')

# Run Integrations
make_observations(runner, looker, number_of_observations, dt, noprog=False)

# Compute Transient Lifetime
transient_lifetime = exit_function(looker.observations)

# Save Exit Time
save_transient_lifetime(transient_lifetime, S, disapearing_attractor)
print(f'Saved transient lifetime {transient_lifetime:.3f} to {save_file}\n')

# Possibly Save Transient - NOT YET FINISHED/TESTED
if save_transients:
    transient_folder = transient_path_folder(S, attractor_transient)
    if not os.path.exists(transient_folder):
        os.makedirs(transient_folder)
    looker.observations.dump(cupboard=transient_folder, name=input_number)
