
import helpers.outils as outils
import numpy as np
import os
import pandas as pd
import warnings

import helpers.outils as outils
import helpers.sap2000 as sap2000

"""
SIMULATION OF TIME HISTORY STRAIN DATA FOR FOOTBRIDGE:

This file obtains a txt with strain time series data from a SAP2000 model.
The output file is analogous to an Arduino Mega recordment by Serial Plot
    (columns: timestamp, Channel 1, Channel 2, ..., Channel n).

OUTPUT FILE:
    filename: name of the output file
    savepath: paths['rawdata_footbridge_simulated_oma']

SAP2000 MODEL:
    - Loadcase: simulated_earthquake:
        Name of the loadcase with time history acceleration data
        IMPORTANT: The loadcase cannot have a very long number of output
            time steps (in my pc, up to 500; in the server, up to 13000).
            Otherwise, the function get_modalforces_timehistory will take
            too long to run.
        IMPORTANT II: The code used for downloading the earthquake is in
            Draft Codes project

Other files required:
    - sg_channels_info.csv: it contains the information of the strain gauges,
        i.e., the locations that are selected as strain outputs in the model.
        In this case, these locations correspond to:
            - Optimal EfI locations for 6 modes and 8 sg
            - Additional 5 locations for a sub-optimal configuration (whose location
                was arbitrary decided by myself)

IN ORDER TO FULLY COMPLETE THIS CODE, IT SHOULD BE DONE:
1) Average modal forces (analogous to EfI algorithm)
2) Check that the coordinates of df_sg are correct (i.e., they correspond to the same
    SAP2000 locations)
"""
# --------------------------------------------------------------
# 1) OPEN AND RUN SAP2000 MODEL
# --------------------------------------------------------------
# 1. Parameters
average_values, round_coordinates = True, True
username = outils.get_username()
sapfile_name = 'Pasarela_v9.sdb'
filename = 'test1.txt'  # output filename
paths = outils.get_paths(os.path.join('src', 'paths', username + '.csv'))

datapath = paths['rawdata_footbridge_simulated_oma']
if filename in os.listdir(datapath):
    old_filename = str(filename)
    while filename in os.listdir(datapath):
        filename = outils.increment_filename(filename)
    warnings.warn(f'{old_filename} already exists. File is renamed to: {filename}')

# 2. Open .sdb model
FilePath = os.path.join(paths['sap2000_footbridge_model'], sapfile_name)
mySapObject = sap2000.app_start()
SapModel = sap2000.open_file(mySapObject, FilePath)

# 3. Set proper units (International System) and run analysis
sap2000.unlock_model(SapModel)
sap2000.set_ISunits(SapModel)
sap2000.run_analysis(SapModel)

# --------------------------------------------------------------
# 2) READ THE DATA FOR SG COORDINATES
# --------------------------------------------------------------
sg_info_name = 'sg_channels_info.csv'
SavePath = os.path.join(datapath, sg_info_name)
df_sg = pd.read_csv(SavePath, sep=';', comment='#')
# to be done: CHECK COORDINATES OF df_sg ARE CORRECT

# --------------------------------------------------------------
# 3) OBTAIN STRAIN TIME HISTORY DATA
# --------------------------------------------------------------
# A) Get modal forces
Name_elements_group = 'modeshape_frames'
loadcase_name = 'simulated_earthquake'
selected_elements = [str(i).split('.')[0] for i in df_sg['SAP2000_id']]
modal_forces, step_time = sap2000.get_modalforces_timehistory(Name_elements_group, SapModel, loadcase_name,
                                                              average_values=average_values,
                                                              round_coordinates=round_coordinates,
                                                              selected_elements=selected_elements)
# to be done: AVERAGE MODAL FORCES

# B) Get section information
Name_points_group, Name_elements_group = "allpoints", "allframes"
all_points, all_elements, all_elements_stat = sap2000.getnames_point_elements(Name_points_group,
                                                                              Name_elements_group,
                                                                              SapModel)
element_section = sap2000.get_elementsections(
    all_elements, all_elements_stat, SapModel)

all_sections = list(set([element_section[i] for i in list(element_section)]))
section_properties_material = sap2000.get_section_information(all_sections,
                                                              SapModel)
# C) Get strain time history
strain_timehistory = outils.get_strain_timehistory(
    modal_forces, element_section, section_properties_material)

active_location = ['up', 'right']  # from  ['right', 'left', 'up', 'down']
num_time_steps = len(step_time)

# D) Get Psi and Psi_id for elements containing strain gauges
Psi, Psi_id = outils.build_Psi_timehistory(strain_timehistory, active_location=active_location,
                                           num_time_steps=num_time_steps)

# E) Filter Psi and Psi_id for stations containing strain gauges
sg_channels_list = list('Element_' + df_sg['SAP2000_id'].astype(str) + '_' + df_sg['SAP2000_dir'])
Psi_id_sg = [i for i, element in enumerate(Psi_id) if element in sg_channels_list]
Psi_sg = Psi[Psi_id_sg, :]

# F) Save data into a txt file with the same structure as ARDUINO
dt = np.mean(np.diff(step_time))
timestamps = np.arange(0, np.shape(Psi_sg)[1] * dt, dt)
channel_columns = [f"Channel {i+1}" for i in range(np.shape(Psi_sg)[0])]
df = pd.DataFrame(np.column_stack((timestamps, Psi_sg.T)), columns=['timestamp'] + channel_columns)
df.to_csv(os.path.join(datapath, filename), sep=',', index=False)

# close SAP2000
mySapObject.ApplicationExit(False)
