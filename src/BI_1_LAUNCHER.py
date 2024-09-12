
import os
import time
import sys

from helpers.outils import get_paths, get_username, load_state, save_state, remove_file
from BI_2_function import run_bayesian_inference

"""
FILE DESCRIPTION:
    Perform Bayesian Inference for a specific FEM model using
    MCMC ALgorithm and OMA data
UPDATABLE PARAMETERS:
    use_starting_parameters: True if the starting parameters are defined in
        algorithm_parameters_filename (so they are not randomly generated)
    load_previous_data: True if the inference was interrupted; False
        for a NEW INFERENCE
    use_GUI: True if SAP2000 GUI is used (only for debugging)
    plot_ConvDiag: True if the convergence diagnostics are plotted
    print_process_MH: True if the Metropolis-Hastings process is printed
    seed: seed for random number generation
    use_threads: True if parallel processing is used for managing SAP2000
        errors (robust but slower) [True is recommended]
    max_retries: maximum number of retries before restarting the full code
COMMENTS:
    run_bayesian_inference function is intended to complete the bayesian
        inference. If an error occurs during the process (due to SAP2000),
        2 possibilities:
            a) The function is restarted
            b) The script is restarted (only after max_retries attempts)
"""
###################################################################
# DATA TO BE UPDATED
###################################################################
# 0. General data
# A) Output Folder Name and Paths
bayesian_inference_name = 'test14_8sg_EFDD'  # Assigned Folder Name
username = get_username()  # File containing paths
algorithm_parameters_filename = 'bayesian_inference_parameters.txt'
paths = get_paths(os.path.join('src', 'paths', username + '.csv'))
algorithm_parameters_path = os.path.join(paths['project'],
                                         'src', 'bayesian_inference_data')
algorithm_output_path = os.path.join(paths['files_tfm_output'],
                                     'bayesian_inference', bayesian_inference_name)

# B) Execution parameters
use_starting_parameters = False  # if True, they must be defined in algorithm_parameters
load_previous_data = False  # False for a NEW INFERENCE
use_GUI = False
plot_ConvDiag = True
print_process_MH = True
seed = 2  # so that we can reproduce all results
use_threads = True  # if True, iterations are slower but process is more robust
max_retries = 3  # maximum number of retries before restarting the full code
###################################################################

###################################################################
# MAIN PROGRAM
###################################################################
# 1. Initializing variables
process_finished = False
retries, iteration_prev = 0, 0

state_filepath = os.path.join(paths['project'], 'state.json')
state = load_state(state_filepath)  # if the process has been interrupted
if state:  # if the process has been interrupted, load previous data
    if not load_previous_data:
        input(f'There is an existing state.json file located in {paths["project"]} which indicates that the inference will continue by loading previous data. If this is not desired, interrupt the execution and delete the file. Press Enter to continue...')
    load_previous_data = state.get('load_previous_data', True)

# 2. Bayesian Inference Process
print('------- Starting Program for Bayesian Inference -------')
while not process_finished:
    process_finished, iteration = run_bayesian_inference(algorithm_parameters_filename, paths,
                                                         algorithm_parameters_path, algorithm_output_path,
                                                         load_previous_data, seed, use_GUI=use_GUI,
                                                         plot_ConvDiag=plot_ConvDiag, print_process_MH=print_process_MH,
                                                         use_threads=use_threads,
                                                         use_starting_parameters=use_starting_parameters)
    if not process_finished:  # Some problem occurs
        retries = retries + 1 if iteration == iteration_prev else 0
        if retries < max_retries:  # If the error occurs for the first time, retry
            print(f"An error occurred. Retry attempt {retries} of {max_retries}. Re-launching the main function.")
            load_previous_data, iteration_prev = True, int(iteration)
            time.sleep(5)
        else:  # restart script
            print(f"An error occurred. Retry attempt {retries} of {max_retries}. Restarting the FULL process.")
            save_state({'load_previous_data': load_previous_data}, state_filepath)
            time.sleep(5)
            os.execv(sys.executable, ['python'] + sys.argv)
    else:  # Process finished successfully
        remove_file(state_filepath)
        print(f'Bayesian Inference Process Completed Successfully; {iteration} iterations completed.')
###################################################################
# END OF FILE
###################################################################
