
import copy
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import threading
import time

import helpers.bayesian_tools as bayes
import helpers.outils as outils
import helpers.plots as plots
import helpers.sap2000 as sap2000

from datetime import datetime


"""
File content:
    Main functions used by the BI_1_LAUNCHER.py file
"""


def timeout_handler(timeout_duration, stop_event) -> None:
    """
    Function Duties:
        Process monitoring; Kills SAP2000 if it exceeds the timeout
    It is defined to be inside a thread
    """
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_duration:
            print("Timeout exceeded, killing SAP2000...")
            process_name = "SAP2000.exe"
            outils.kill_process_advanced(process_name)
            stop_event.set()  # stop_event is set to True
            break
        time.sleep(0.05)  # Verify every 0.05 seconds


def run_bayesian_inference(algorithm_parameters_filename, paths,
                           algorithm_parameters_path, algorithm_output_path,
                           load_previous_data, seed, use_GUI=False, plot_ConvDiag=False,
                           print_process_MH=False, use_threads=False,
                           use_starting_parameters=False):
    """
    Function Duties:
        Perform Bayesian Inference for a specific FEM model using
            MCMC ALgorithm and OMA data
        See bayesian_inference_launcher.py for further details
        Files updated in each iteration:
            h5py file with the following data:
                - parameters: parameters used in the analysis
                - frequencies: frequencies obtained from the analysis
                - phi: displacement modeshapes
                - psi: strain modeshapes
                - LN_Lik_prop: log likelihood
                - accepted: boolean indicating if the sample was accepted
            process.txt: text file with the process of the algorithm
            errors.txt: text file with the errors generated during the process
        Additionally saves fem_parameters.json: parameters used in the FEM model
    Input:
        algorithm_parameters_filename: str
        paths: dict
        algorithm_parameters_path: str
        algorithm_output_path: str
        load_previous_data: bool
        use_GUI: bool
        plot_ConvDiag: bool
        print_process_MH: bool
        seed: int
    Return:
        Success: bool
        i: iteration number
    """
    # 0. Kill SAP2000 if it is being runned (otherwise raises error)
    process_name = "SAP2000.exe"
    outils.kill_process_advanced(process_name)

    # 1. Read parameters and configure additional variables
    if load_previous_data:
        parameters_filepath = os.path.join(algorithm_output_path, algorithm_parameters_filename)
    else:
        parameters_filepath = os.path.join(algorithm_parameters_path, algorithm_parameters_filename)
    algorithm_parameters = outils.read_parameters(parameters_filepath)

    if use_GUI:
        SolverProcessType = 1
    else:
        SolverProcessType = 2

    np.random.seed(seed)

    # 2. Load experimental OMA data (obtained with oma.py file)
    FilePath = os.path.join(
        paths['oma_results'], algorithm_parameters['measured_oma_filename'])
    with open(FilePath, 'r') as json_file:
        experimental_modal = json.load(json_file)
    frequencies_oma = experimental_modal['frequencies']
    Psi_oma_real = np.array(experimental_modal['real_strain_modeshapes'])
    Psi_oma_id = experimental_modal['channels_id']

    # 3. Prepare data for the Algorithm
    # 3.1 Load previous data
    if load_previous_data:
        # 3.1 Bayesian Inference Data
        h5py_file = os.path.join(
            algorithm_output_path, algorithm_parameters['bi_output_filename'])
        groups = ['parameters', 'frequencies',
                  'phi', 'psi', 'LN_Lik_prop', 'accepted']
        matrices = outils.read_matrices_h5py(h5py_file, groups)
        parameters, frequencies = matrices['parameters'], matrices['frequencies']
        phi, psi = matrices['phi'], matrices['psi']
        LN_Lik_prop, accepted = matrices['LN_Lik_prop'], [
            bool(i[0, 0]) for i in matrices['accepted']]
        # 3.2 FEM Data
        with open(os.path.join(algorithm_output_path,
                               algorithm_parameters['fem_parameters_filename']), 'r') as json_file:
            fem_parameters = json.load(json_file)
        Phi_id_raw = fem_parameters['Phi_id_raw']
        Psi_id_raw = fem_parameters['Psi_id_raw']
        Psi_id = copy.deepcopy(Psi_oma_id)
    else:
        now = datetime.now()
        date_time_str = now.strftime('%Y%m%d_%H%M')
        backup_path = os.path.join(algorithm_output_path, f'backup_{date_time_str}')
        os.makedirs(backup_path, exist_ok=True)
        for file in [algorithm_parameters['bi_output_filename'], algorithm_parameters['errors_filename'],
                     algorithm_parameters['process_filename'], algorithm_parameters['fem_parameters_filename'],
                     algorithm_parameters_filename]:
            file_path = os.path.join(algorithm_output_path, file)
            if file in os.listdir(algorithm_output_path):
                if file in os.listdir(backup_path):
                    os.remove(os.path.join(backup_path, file))
                shutil.move(file_path, backup_path)

    # 3.2 Get parameters limits in proper format
    Definition_interv = bayes.generate_def_interv(algorithm_parameters['E_interv'], algorithm_parameters['rho_interv'],
                                                  algorithm_parameters['R1_interv'], algorithm_parameters['R2_interv'],
                                                  algorithm_parameters['T_interv'], algorithm_parameters['M2_interv'],
                                                  algorithm_parameters['M3_interv'],
                                                  algorithm_parameters['sigma_f_interv'],
                                                  algorithm_parameters['w_interv'],
                                                  n_frames=algorithm_parameters['n_frames'],
                                                  n_joints=algorithm_parameters['n_joints'])

    if use_starting_parameters and not load_previous_data:
        Definition_ini = bayes.generate_def_interv(algorithm_parameters['E_ini'], algorithm_parameters['rho_ini'],
                                                   algorithm_parameters['R1_ini'], algorithm_parameters['R2_ini'],
                                                   algorithm_parameters['T_ini'], algorithm_parameters['M2_ini'],
                                                   algorithm_parameters['M3_ini'],
                                                   algorithm_parameters['sigma_f_ini'],
                                                   algorithm_parameters['w_ini'],
                                                   n_frames=algorithm_parameters['n_frames'],
                                                   n_joints=algorithm_parameters['n_joints'])
        Definition_ini = {key: np.array(Definition_ini[key]) for key in Definition_ini}
        Definition_ini = bayes.check_ini_values(Definition_ini, Definition_interv)

    # 3.3 Other MCMC parameters
    parameters_groups = list()
    if (algorithm_parameters['E_interv'] or algorithm_parameters['rho_interv']):
        parameters_groups += ['material']
    if algorithm_parameters['R1_interv'] or algorithm_parameters['R2_interv']:
        parameters_groups += ['joint_springs']
    if algorithm_parameters['T_interv'] or algorithm_parameters['M2_interv'] or algorithm_parameters['M3_interv']:
        parameters_groups += ['frame_releases']
    niters = algorithm_parameters['niters']
    WhereConvTest = np.array(
        [x**3 for x in np.arange(1, niters**(1/3), algorithm_parameters['jmp'])])

    # 4. Open SAP2000, unlock model and set proper units
    try:
        # First we delete the file to avoid issues due to a bad closing
        FilePath = os.path.join(
            paths['project'], 'src', 'sap2000', algorithm_parameters['sap2000_filename'])
        try:
            os.remove(FilePath)
        except:
            pass
        # Re-copy the file and open it
        FilePath_backup = os.path.join(paths['project'], 'src', 'sap2000', 'backup_uncalibrated',
                                       algorithm_parameters['sap2000_filename'])
        shutil.copy(FilePath_backup, FilePath)
        mySapObject = sap2000.app_start(use_GUI)
        SapModel = sap2000.open_file(mySapObject, FilePath)
        sap2000.unlock_model(SapModel)
        sap2000.set_solver(SapModel, SolverType=2, SolverProcessType=SolverProcessType,
                           NumberParallelRuns=0, ResponseFileSizeMaxMB=-1, NumberAnalysisThreads=-1,
                           StiffCase="MODAL")
        sap2000.set_ISunits(SapModel)
    except Exception as e:
        errors_filepath = os.path.join(
            algorithm_output_path, algorithm_parameters['errors_filename'])
        with open(errors_filepath, 'a') as file:
            file.write(f'Error before loop: {e}\n')
        Success = False
        return Success, 0

    #######################################
    # METROPOLIS-HASTING
    #######################################
    # 2. Algorithm parameters
    # A) Updatable parameters
    # Updatable parameters (all in this case)
    Testing = list(Definition_interv)
    # stdev of the proposal
    PercentOnInterval = algorithm_parameters['proposal_stdev'] * \
        np.ones(len(Testing))

    # B) Variables to allocate results
    if load_previous_data:
        starting_i = np.where(np.any(frequencies != 0, axis=(1, 2)))[0][-1] + 1
        Theta_computed = [list(parameters[i, :, 0]) for i in range(starting_i)]
        Theta_remaining = [[] for i in range(starting_i, niters+1)]
        Theta = Theta_computed + Theta_remaining
        Targt_computed = np.array([LN_Lik_prop[i, 0, 0]
                                  for i in range(starting_i)]).reshape(starting_i)
        Targt_remaining = np.zeros(niters+1-starting_i)
        Targt = np.concatenate((Targt_computed, Targt_remaining))
        accepted = [i for i in accepted[:starting_i]]
        cumulative_sum = np.cumsum(accepted[:starting_i])
        acc_rate_computed = cumulative_sum / np.arange(1, starting_i + 1)
        acc_rate_remaining = np.zeros(niters+1-starting_i)
        acc_rate = np.concatenate((acc_rate_computed, acc_rate_remaining))
        naccept = np.sum(accepted)
    else:
        starting_i, accepted = 0, list()
        Theta = [[] for i in range(niters+1)]  # parameters
        acc_rate = np.zeros(niters+1)
        Targt = np.zeros(niters+1)
        naccept = 0

    # 3. Run algorithm
    bi_output_filepath = os.path.join(
        algorithm_output_path, algorithm_parameters['bi_output_filename'])
    opening_mode = 'w' if not load_previous_data else 'r+'
    with h5py.File(bi_output_filepath, opening_mode) as h5py_file:
        warning_dict = dict()
        for i in range(starting_i, niters+1):
            # 1. Generate parameters
            start_time = time.time()  # Record the start time
            if i == 0:  # initial case: start by initial parameters
                Theta_prop, ME = bayes.initial_parameters(Definition_interv)
                if use_starting_parameters:  # only parameters whose ini values are specified are updated
                    Theta_prop.update(Definition_ini)
                Parameters = copy.deepcopy(Theta_prop)
                warn_dict = dict()
            else:  # sample from normal distribution with interval restrictions
                Theta_prop, ME, warn_dict = bayes.proposal(
                    Theta[i-1], Definition_interv, PercentOnInterval, Testing)
                if i == starting_i and load_previous_data:
                    Parameters = copy.deepcopy(Theta_prop)
            Parameters.update(Theta_prop)
            if len(warn_dict) > 0:
                warning_dict[i] = dict()
                warning_dict[i]['proposal'] = warn_dict
            input_data = outils.from_parameters_to_input(Parameters, material_name='frame-Steel',
                                                         joint_name='supportpoints', frame_name='releaseframes',
                                                         parameters_groups=parameters_groups)

            # 1.1 Start timer to kill SAP2000 if it exceeds the timeout
            if use_threads:
                stop_event = threading.Event()
                timeout_duration = 60*2  # 2 minutes allowed to work
                timer_thread = threading.Thread(target=timeout_handler,
                                                args=(timeout_duration, stop_event))
                timer_thread.start()

            # 2. Run analysis and get modal response
            try:
                # Run SAP2000 (which might get stuck)
                modal_results = sap2000.get_modal_response(input_data, SapModel)

                # Stop the timer_thread as everything worked properly
                if use_threads:
                    stop_event.set()  # so that timer_thread stops
                    timer_thread.join()  # wait until the thread finishes

            except Exception as e:
                # Stop the timer_thread before restarting
                if use_threads:
                    stop_event.set()  # so that timer_thread stops
                    timer_thread.join()  # wait until the thread finishes
                ME = True
                errors_filepath = os.path.join(
                    algorithm_output_path, algorithm_parameters['errors_filename'])
                with open(errors_filepath, 'a') as file:
                    parameters_string = ', '.join(
                        [f"{key}={value}" for key, value in Parameters.items()])
                    file.write(f'Iteration {i}: {e}; {parameters_string}\n')

                Success = False

                return Success, i

            # A) Arrange modal results
            frequencies, disp_modeshapes, strain_modeshapes = dict(), dict(), dict()
            for mode in list(modal_results):
                frequencies[mode] = modal_results[mode]['frequencies']['Frequency']
                disp_modeshapes[mode] = copy.deepcopy(
                    modal_results[mode]['disp_modeshapes'])
                strain_modeshapes[mode] = copy.deepcopy(
                    modal_results[mode]['strain_modeshapes'])

            # B) Get displacement modeshapes for ALL CHANNELS (to be further analyzed)
            active_dofs = ['U1', 'U2']
            Phi_raw, Phi_id_raw = outils.build_Phi(
                disp_modeshapes, active_dofs)

            # C) Get the strain modeshapes for SELECTED channels
            active_location = list(
                set(i.split('_')[-1] for i in Psi_oma_id))
            Psi_raw, Psi_id_raw = outils.build_Psi(strain_modeshapes,
                                                   active_location=active_location)
            Psi, Psi_id = outils.Psi_selected_dofs(
                Psi_raw, Psi_id_raw, Psi_oma_id)
            if Psi_id != Psi_oma_id:
                raise ValueError(
                    'Strain gauges are not the same in both models')

            # D) Mode matching
            frequencies_matched, Psi_matched, mode_matching = outils.mode_matching_mac(
                frequencies, Psi, Psi_oma_real)

            # 3. Compute log likelihood
            frequencies_oma_values = [frequencies_oma[i] for i in frequencies_oma]
            LN_Lik_prop, ME = bayes.Log_likelihood(Parameters, frequencies_matched,
                                                   frequencies_oma_values,
                                                   Psi_matched, Psi_oma_real, ME)

            if ME or np.isinf(LN_Lik_prop):
                if i == 0:
                    Theta[i], Targt[i] = copy.deepcopy(
                        Theta_prop), copy.deepcopy(LN_Lik_prop)
                else:
                    Theta[i], Targt[i] = Theta[i-1], Targt[i-1]
                    # i+1 because i=0 in first iteration
                    acc_rate[i] = naccept/(i+1)
                    accepted.append(False)
                    continue

            # 4. Compute ratio of likelihoods
            Targt_prop = copy.deepcopy(LN_Lik_prop)
            LN_alfa = min(0, Targt_prop - Targt[i-1])  # ratio of likelihoods

            # 5. Accept / reject sample
            if i == 0:  # Save results if first iteration
                Theta[i] = [Theta_prop[Testing[j]]
                            for j in range(len(Testing))]
                Targt[i] = copy.deepcopy(Targt_prop)
                # assuming Gaussian prop. and unif. prior
                if np.exp(LN_alfa) >= np.random.rand():
                    naccept += 1
                    accepted.append(True)
                else:
                    accepted.append(False)
            else:  # Save proposed parameters
                # assuming Gaussian prop. and unif. prior
                if np.exp(LN_alfa) >= np.random.rand():
                    Theta[i] = [Theta_prop[Testing[j]]
                                for j in range(len(Testing))]
                    Targt[i] = copy.deepcopy(Targt_prop)
                    naccept += 1
                    accepted.append(True)
                else:  # Do not save proposed parameters
                    Theta[i], Targt[i] = copy.deepcopy(
                        Theta[i-1]), copy.deepcopy(Targt[i-1])
                    accepted.append(False)

            acc_rate[i] = naccept/(i+1)

            # 6. Analysis
            if (i == WhereConvTest).any():  # convergence diagnostics
                if plot_ConvDiag:
                    fig, ax = plots.plot_ConvDiag(Targt[0:i], acc_rate[0:i])

                    SavePath = os.path.join(paths['savefigures'], algorithm_output_path.split(os.sep)[-1])
                    figname = f'MCMC_Convergence_{i}_iterations.pdf'
                    if not os.path.isdir(SavePath):
                        os.makedirs(SavePath)  # if file does not exist, creates it
                    fig.savefig(os.path.join(SavePath, figname), format='pdf', dpi='figure')
                    plt.close('all')

            # 7. Save in HDF5 and fem_parameters (if first iteration)
            parameters = np.array([float(Parameters[key])
                                  for key in Parameters])
            output_frequencies = np.array(
                [frequencies[id] for id in frequencies])
            # A) Allocate values for first iteration
            if i == 0 and not load_previous_data:
                # Preallocate space for the datasets
                names = ['parameters', 'frequencies',
                         'phi', 'psi', 'LN_Lik_prop', 'accepted']
                dimensions = [(niters+1, len(parameters), 1), (niters+1, len(output_frequencies), 1),
                              (niters+1, np.shape(Phi_raw)
                               [0], np.shape(Phi_raw)[1]),
                              (niters+1, np.shape(Psi_raw)
                               [0], np.shape(Psi_raw)[1]),
                              (niters+1, 1, 1), (niters+1, 1, 1)]
                for name, dim in zip(names, dimensions):
                    h5py_file.create_dataset(name, dim, dtype='float64')

                # Save bayesian_inference info json
                fem_parameters = {
                    'Phi_id_raw': Phi_id_raw, 'Psi_id_raw': Psi_id_raw}
                fem_parameters_filepath = os.path.join(
                    algorithm_output_path, algorithm_parameters['fem_parameters_filename'])
                with open(fem_parameters_filepath, 'w') as json_file:
                    json.dump(fem_parameters, json_file)

                # Save algorithm parameters
                parameters_destination_filepath = os.path.join(
                    algorithm_output_path, algorithm_parameters_filename)
                shutil.copy(parameters_filepath, parameters_destination_filepath)

            # B) Save data
            # b.1 Input data
            h5py_file['parameters'][i, :, 0] = parameters
            # b.2 FEM output data
            h5py_file['frequencies'][i, :, 0] = output_frequencies
            h5py_file['phi'][i, :, :] = Phi_raw
            h5py_file['psi'][i, :, :] = Psi_raw
            # b.3 bayesian inference data
            h5py_file['LN_Lik_prop'][i, 0, 0] = LN_Lik_prop
            h5py_file['accepted'][i, 0, 0] = accepted[-1]

            # 8. Save process
            if print_process_MH:
                if np.mod(i, 1) == 0:
                    print(f'Iteration {i+1}/{niters}')
            end_time = time.time()
            time_required = end_time - start_time
            current_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            process_filepath = os.path.join(
                algorithm_output_path, algorithm_parameters['process_filename'])
            with open(process_filepath, "a") as file:
                text = f"Iteration {i}: time required: {time_required:.2f} seconds, endtime: {current_time}\n"
                file.write(text)

    mySapObject.ApplicationExit(False)
    Success = True

    return Success, i
