
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

import helpers.outils as outils
import helpers.sap2000 as sap2000
import helpers.plots as plots

"""
File Description:
    EFI Algorithm for Sensor Placement

Created on 2024/04/19 by asanchezlc

Description:
The file contains 2 parts:
- Part 1: SAP2000 Calculation
    - Modal frequencies
    - Displacement mode shapes
    - Strain mode shapes
- Part 2: EFI Algorithm
    - Accelerometers (using displacement mode shapes)
    - Strain gauges (using strain mode shapes)
- Part 3: Plots
    - Remark: mode shape plots only available for test_beam.sdb

Two sap2000 files are available:
    - test_beam.sdb: simple beam model
    - uncalibrated.sdb: frame model  
"""
"""
PART 1: SAP2000 CALCULATION

SAP2000 model is run in order to obtain modal information:
- Modal frequencies
- Displacement mode shapes
- Strain mode shapes
This information is properly stored to be used in the EFI algorithm.

Parameters' information:
- average_values: if True, modal forces are averaged in the same point
    Note that if two elements arrive to a point, that point will have two different
    values; averaging avoid it. If false, left-hand value is taken
- round_coordinates: if True, coordinates are rounded to 6 significant digits
    Note that some times a value of 0.0 is read from SAP2000 as 0.0000000000001
- username: one of the specified in paths folder (useful for managin different paths
    when cloning the repository)
- sapfile_name: any SAP file stored in sap2000 folder.
    Note that any of those SAP files must have the following groups defined:
    - allpoints: all points in the model
    - allframes: all frames in the model
    - modeshape_points: points where displacement mode shapes are retrieved
    - modeshape_frames: frames where strain mode shapes are retrieved
    - nodenotselectable: points that cannot be selected as sensors
    - framesnotselectable: frames that cannot be selected as sensors
"""
# --------------------------------------------------------------
# 1) OPEN AND RUN SAP2000 MODEL
# --------------------------------------------------------------
# 1. Parameters
average_values, round_coordinates = True, True
username = outils.get_username()
# sapfile_name = 'test_beam.sdb'
sapfile_name = 'uncalibrated.sdb'  # 'frame.sdb'

# 2. Open .sdb and unlock the model
paths = outils.get_paths(os.path.join('src', 'paths', username + '.csv'))
FilePath = os.path.join(paths['project'], 'src', 'sap2000', sapfile_name)
mySapObject = sap2000.app_start()
SapModel = sap2000.open_file(mySapObject, FilePath)
sap2000.unlock_model(SapModel)

# 3. Set proper units (International System)
sap2000.set_ISunits(SapModel)

# 4. Run Analysis
sap2000.run_analysis(SapModel)

# --------------------------------------------------------------
# 2) OBTAIN EIGENFREQUENCIES AND DISPLACEMENT MODE SHAPES
# --------------------------------------------------------------
# 1. Modal frequencies
frequencies = sap2000.get_modalfrequencies(SapModel)

# 2. Displacement mode shapes
Name_points_group = "modeshape_points"
disp_modeshapes = sap2000.get_displmodeshapes(Name_points_group, SapModel)

# --------------------------------------------------------------
# 3) OBTAIN STRAIN MODE SHAPES
# --------------------------------------------------------------
# 1. Prepare data for obtaining strain mode shapes
# A) Get elements
Name_points_group, Name_elements_group = "allpoints", ""
all_points, _, _ = sap2000.getnames_point_elements(Name_points_group, Name_elements_group,
                                                   SapModel)
joint_coordinates = sap2000.get_pointcoordinates(all_points, SapModel)

# A) Get elements
Name_points_group, Name_elements_group = "allpoints", "allframes"
all_points, all_elements, all_elements_stat = sap2000.getnames_point_elements(Name_points_group,
                                                                              Name_elements_group,
                                                                              SapModel)
# B) Get section assigned to each element
element_section = sap2000.get_elementsections(
    all_elements, all_elements_stat, SapModel)

# C) Get section information
all_sections = list(set([element_section[i] for i in list(element_section)]))
section_properties_material = sap2000.get_section_information(all_sections,
                                                              SapModel)
# D) Get modal forces (from which strain modes are calculated)
Name_elements_group = 'modeshape_frames'
modal_forces = sap2000.get_modalforces(Name_elements_group, SapModel,
                                       average_values=average_values, round_coordinates=round_coordinates)

# E) Get information about point and element coordinates (important for averaging and for plotting)
all_points_coord = sap2000.get_pointcoordinates(
    all_points, SapModel, round_coordinates=round_coordinates)
all_elements_coord_connect = sap2000.get_frameconnectivity(all_points, all_elements,
                                                           SapModel, all_points_coord=all_points_coord)

# F) Average modal forces (if desired)
if average_values:
    all_points_connect_elements = outils.get_pointconnectivity(
        all_points_coord, all_elements_coord_connect)
    modal_forces = outils.average_forces(
        all_points_connect_elements, modal_forces)

# G) Get strain mode shapes
strain_modeshapes = outils.get_strainmodeshapes(
    modal_forces, element_section, section_properties_material)

# H) Get coordinates from points in which strain modes are retrieved
elements_id_coord = outils.get_element_mesh_coordinates(all_elements_coord_connect, strain_modeshapes,
                                                        round_coordinates=round_coordinates)

# --------------------------------------------------------------
# 4) SAVE RESULTS
# --------------------------------------------------------------
# 1. Merge all results into a single dictionary
all_dict = {'frequencies': frequencies,
            'disp_modeshapes': disp_modeshapes,
            'strain_modeshapes': strain_modeshapes}

modal_results = outils.merge_all_dict(all_dict)

# 2. Save data
# Modal results
SavePath = os.path.join(paths['project'], 'src',
                        'EfI_data', 'modal_results.json')
with open(SavePath, 'w') as json_file:
    json.dump(modal_results, json_file)

# Joint information
SavePath = os.path.join(paths['project'], 'src',
                        'EfI_data', 'joint_coordinates.json')
with open(SavePath, 'w') as json_file:
    json.dump(joint_coordinates, json_file)

# Element information
SavePath = os.path.join(paths['project'], 'src',
                        'EfI_data', 'element_coordinates.json')
with open(SavePath, 'w') as json_file:
    json.dump(elements_id_coord, json_file)
"""
PART 2: EFI ALGORITHM

EFI Algorithm is run in order to obtain the optimal sensor placement
for:
- Accelerometers (using displacement mode shapes)
- Strain gauges (using strain mode shapes)
Results are plotted and saved.

Parameters' information:
- active_location: location of the SG in the beam: the convention
    is the following: up refers to the upper part of the beam when
    displayed in SAP2000 (define -> section properties), i.e., the
    top of the "2" local axis. It is in the 1-3 plane.
    Analogous for right, left, and down.
- modes_considered: modes to be considered in the EFI algorithm
    (previously seen in SAP2000)
"""
# --------------------------------------------------------------
# 1) EFI ALGORITHM FOR STRAIN MODE SHAPES
# --------------------------------------------------------------
# 0. Algorithm variables
filename_SG = 'EFI_SG'  # for saving
if sapfile_name == 'test_beam.sdb':
    active_location = ['up', 'right']  # from  ['right', 'left', 'up', 'down']
    n_sensors_sg = 10
    modes_considered_sg = list(np.arange(0, len(strain_modeshapes)))
    # modes_considered_sg = [1, 4, 8]
elif sapfile_name == 'uncalibrated.sdb':
    active_location = ['up']
    # modes_considered_sg = [0, 2, 3, 4, 5, 6, 8, 10]  # 8 modes
    # modes_considered_sg = [0, 2, 3, 4, 5, 6, 8]  # 7 modes
    modes_considered_sg = [0, 2, 3, 4, 5, 6]  # 6 target modes
    n_sensors_sg = 6

# 1. Delete slave positions (restrained + not selectable)
# TO BE DONE

# 1.2 Build Phi and delete restraints and not_selectable
Psi, Psi_id = outils.build_Psi(strain_modeshapes,
                               active_location=active_location)
Psi = Psi[:, modes_considered_sg]

# DELETE SLAVES-> TO BE DONE

if n_sensors_sg < np.shape(Psi)[1]:
    warnings.warn('Number of modes will become indistinguishable')

# 2. EFI Algorithm
EFI_results_SG = outils.EFI(Psi, Psi_id, n_sensors_sg)
Psi, Psi_id = np.array(EFI_results_SG['results']['Phi']), np.array(
    EFI_results_SG['results']['Phi_id'])
detFIM_SG, Ed_all_SG = EFI_results_SG['process']['detFIM'], EFI_results_SG['process']['Ed_all']
deleted_channels_SG = EFI_results_SG['process']['deleted_channels']

# 3. Plot sensors in SAP2000 model
points_for_plotting = outils.prepare_dict_plotting_SG(
    Psi_id, strain_modeshapes)
sap2000.plot_SG_as_forces(Psi_id, points_for_plotting, len(modes_considered_sg), SapModel)

# 4. Save results
SavePath = os.path.join(paths['project'], 'src',
                        'EfI_data', filename_SG + '.json')
with open(SavePath, 'w') as json_file:
    json.dump(EFI_results_SG, json_file)

# Print and save SG Location
df_sg = pd.DataFrame(columns=['SG_id', 'x', 'y', 'z', 'dir'])
df_sg['SG_id'] = [i.split('_')[0] for i in Psi_id]
df_sg['dir'] = [i.split('_')[1] for i in Psi_id]
for i, loc in enumerate(df_sg['SG_id']):
    df_sg.loc[df_sg['SG_id'] == loc, 'x'] = elements_id_coord[loc]['x']
    df_sg.loc[df_sg['SG_id'] == loc, 'y'] = elements_id_coord[loc]['y']
    df_sg.loc[df_sg['SG_id'] == loc, 'z'] = elements_id_coord[loc]['z']
print(df_sg)
filename_SG_loc = f'SG_location_{len(modes_considered_sg)}_modes_{n_sensors_sg}_channels.csv'
SavePath = os.path.join(paths['project'], 'src',
                        'EfI_data', filename_SG_loc)
df_sg.to_csv(SavePath, index=False)

# --------------------------------------------------------------
# 2) EFI ALGORITHM FOR DISPLACEMENT MODE SHAPES
# --------------------------------------------------------------
# 0. Algorithm variables
filename_accel = 'EFI_accel'  # for saving
active_dofs = ['U1', 'U2', 'U3']
n_sensors_accel = 10
modes_considered_accel = list(np.arange(0, len(disp_modeshapes)))
# modes_considered_accel = [0, 2, 3, 5, 7, 9]

# 1. Delete slave positions (restrained + not selectable)
# 1.1 Retrieve joints
candidates = outils.sort_list_string(list(joint_coordinates))
restraints = sap2000.get_pointrestraints(candidates, SapModel)

Name_points_group, Name_elements_group = "nodenotselectable", ""
not_selectable, _, _ = sap2000.getnames_point_elements(
    Name_points_group, Name_elements_group, SapModel)

# 1.2 Build Phi and delete restraints and not_selectable
Phi, Phi_id = outils.build_Phi(
    disp_modeshapes, active_dofs=active_dofs)
Phi, Phi_id = outils.Phi_delete_slaves(Phi, Phi_id, restraints, not_selectable)
Phi = Phi[:, modes_considered_accel]

if n_sensors_accel < np.shape(Phi)[1]:
    warnings.warn('Number of modes will become indistinguishable')

# 2. EFI Algorithm
EFI_results_accel = outils.EFI(Phi, Phi_id, n_sensors_accel)
Phi, Phi_id = np.array(EFI_results_accel['results']['Phi']), np.array(
    EFI_results_accel['results']['Phi_id'])
detFIM_accel, Ed_all_accel = EFI_results_accel['process']['detFIM'], EFI_results_accel['process']['Ed_all']
deleted_channels_accel = EFI_results_accel['process']['deleted_channels']

# 3. Plot sensors in SAP2000 model (and run again to let SAP2000 model prepared)
sap2000.plot_accelerometers_as_forces(Phi_id, len(modes_considered_accel), SapModel)
# sap2000.run_analysis(SapModel)

# 4.Save results
SavePath = os.path.join(paths['project'], 'src',
                        'EfI_data', filename_accel + '.json')
with open(SavePath, 'w') as json_file:
    json.dump(EFI_results_accel, json_file)

# Print and save accelerations Location
df_accel = pd.DataFrame(columns=['accel_id', 'x', 'y', 'z', 'dir'])
df_accel['accel_id'] = [i.split('_')[0] for i in Phi_id]
df_accel['dir'] = [i.split('_')[1] for i in Phi_id]
for i, loc in enumerate(df_accel['accel_id']):
    df_accel.loc[df_accel['accel_id'] == loc,
                 'x'] = joint_coordinates[loc]['x']
    df_accel.loc[df_accel['accel_id'] == loc,
                 'y'] = joint_coordinates[loc]['y']
    df_accel.loc[df_accel['accel_id'] == loc,
                 'z'] = joint_coordinates[loc]['z']
print(df_accel)
filename_accel_loc = f'accel_location_{len(modes_considered_accel)}_modes_{n_sensors_accel}_channels.csv'
SavePath = os.path.join(paths['project'], 'src',
                        'EfI_data', filename_accel_loc)
df_accel.to_csv(SavePath, index=False)

"""
PART 3: DRAFT: MODE SHAPE PLOTS and others

EFI Algorithm is run in order to obtain the optimal sensor placement
for:
- Accelerometers (using displacement mode shapes)
- Strain gauges (using strain mode shapes)
Results are plotted and saved.
"""
# A) Ed and FIM plots
savefigures_path = os.path.join(paths['savefigures'], 'EFI')
if not os.path.exists(savefigures_path):
    os.makedirs(savefigures_path)

for Ed_all, detFIM, title, n_sensors, n_modes in zip([Ed_all_SG, Ed_all_accel],
                                                     [detFIM_SG, detFIM_accel],
                                                     ['Strain Gauges', 'Accel'],
                                                     [n_sensors_sg, n_sensors_accel],
                                                     [len(modes_considered_sg), len(modes_considered_accel)]):
    Ed_removed = [min(i) for i in Ed_all]
    fig, ax = plots.EFI_evolution(Ed_removed, detFIM, n_sensors, n_modes, title)
    fig_name = f'EFI_{title.replace(" ", "")}_Algorithm_Evol_{n_modes}_modes_{n_sensors}_channels.png'
    fig.savefig(os.path.join(savefigures_path, fig_name))
    fig, ax = plots.EFI_evolution_simplified(Ed_removed, detFIM, n_sensors, n_modes, title)
    fig_name = f'EFI_{title.replace(" ", "")}_Algorithm_Evol_{n_modes}_modes_{n_sensors}_channels_bis.pdf'
    fig.savefig(os.path.join(savefigures_path, fig_name))

# B) DISPLACEMENT MODE SHAPES (ONLY FOR test_beam.sdb)
if sapfile_name == 'test_beam.sdb':
    mode_type = ['T', 'V', 'T', 'T', 'V', 'T', 'L', 'T', 'V', 'T']
# elif sapfile_name == 'uncalibrated.sdb':
#     mode_type = ['x', 'y', 't', 'x', 'x', 'x', 'x', 'y', 't', 'y', 't', 'y']

if sapfile_name == 'test_beam.sdb':
    joint_saved = [i.split('_')[0] for i in list(Phi_id)]
    dof_saved = [i.split('_')[1] for i in list(Phi_id)]
    x_coord_accel = [joint_coordinates[i]['x'] for i in joint_saved]

    for n_mode in range(np.shape(Phi)[1]):
        mode_label = list(disp_modeshapes)[n_mode]
        if mode_type[n_mode] == 'V':
            U_label, ylabel = 'U3', 'Z'
        elif mode_type[n_mode] == 'T':
            U_label, ylabel = 'U2', 'Y'
        else:
            U_label, ylabel = 'U1', 'X'
        # SAP2000 Displacement Mode
        x_coord = np.array([joint_coordinates[i]['x']
                        for i in list(joint_coordinates)])
        y_coord = np.array(disp_modeshapes[mode_label][U_label])
        id_sort = np.argsort(x_coord)
        x_coord, y_coord = x_coord[id_sort], y_coord[id_sort]

        # SENSORS Displacement Mode
        bool_mode = list(np.array(dof_saved) == U_label)
        x_accel = np.array(x_coord_accel)[bool_mode]
        y_accel = Phi[bool_mode, n_mode]

        # Plotting
        fig, ax = plt.subplots(1)
        ax.plot(x_coord, y_coord)
        ax.scatter(x_accel, y_accel)
        ax.set_xlabel('X [m]')
        ax.set_ylabel(ylabel)
        ax.set_title('Mode shape ' + str(n_mode+1))
        ax.grid()

    element_saved = [i.split('_')[0] for i in list(Psi_id)]
    location_saved = [i.split('_')[1] for i in list(Psi_id)]

    # C) STRAIN MODE SHAPES (ONLY FOR test_beam.sdb)
    for n_mode in range(np.shape(Psi)[1]):
        # SAP2000 Strain Mode
        mode_label = list(strain_modeshapes)[n_mode]
        if mode_type[n_mode] == 'V' or mode_type[n_mode] == 'x' or mode_type[n_mode] == 't':
            eps_aux = [strain_modeshapes[mode_label][i]['epsilon_1_3_up']
                    for i in list(strain_modeshapes[mode_label])]
            ylabel = 'Beam Top'
        else:
            eps_aux = [strain_modeshapes[mode_label][i]['epsilon_1_2_right']
                    for i in list(strain_modeshapes[mode_label])]
            ylabel = 'Beam Right Face'
        id_aux = [strain_modeshapes[mode_label][i]['Mesh_id']
                for i in list(strain_modeshapes[mode_label])]
        eps = list()
        for i in eps_aux:
            eps += i
        id_elements = list()
        for i in id_aux:
            id_elements += i
        x_coord = [elements_id_coord[i]['x'] for i in id_elements]

        # SENSORS Strain Mode
        x_coord_sg = [elements_id_coord[i]['x'] for i in element_saved]
        y_coord_sg = Psi[:, n_mode]
        if mode_type[n_mode] == 'V':
            bool_mode = list(np.array(location_saved) == 'up')
        else:
            bool_mode = list(np.array(location_saved) == 'right')
        x_coord_sg, y_coord_sg = np.array(
            x_coord_sg)[bool_mode], np.array(y_coord_sg)[bool_mode]

        # Plotting
        fig, ax = plt.subplots(1)
        ax.plot(x_coord, eps)
        ax.scatter(x_coord_sg, y_coord_sg)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('Strain ' + ylabel)
        ax.set_title('Strain mode shape: ' + mode_label)
        ax.grid()

# element_id = [i.split('_')[0] for i in list(Psi_id)]
# coord = dict()
# for element in element_id:
#     coord[element] = elements_id_coord[element]


# CODE FOR OBTAINING THE ID FROM ANY ARBITRARY POINT
# x, y, z = 0.5, 0, 0.6625
# for element in elements_id_coord:
#     coord = elements_id_coord[element]
#     if coord['x'] == x and coord['y'] == y and coord['z'] == z:
#         print(element)
#         break

# CODE FOR OBTAINING THE ID FROM ANY ARBITRARY POINT FOR ACCELEROMETERS
# folder = r'C:\Users\Usuario\Documents\DOCTORADO_CODES\ARDUINO\AccelerometersEGMFrame\AZDelivery\frame_records\oma'
# file = 'acc_channels_info.csv'
# df = pd.read_csv(os.path.join(folder, file), sep=';')
# for i in range(np.shape(df)[0]):
#     df_ch = df.iloc[i]
#     x, y, z = float(df_ch['x']), float(df_ch['y']), float(df_ch['z'])
#     for element in joint_coordinates:
#         coord = joint_coordinates[element]
#         if coord['x'] == x and coord['y'] == y and coord['z'] == z:
#             df.iloc[i, 4] = int(element)
#             break
# df['SAP2000_id'] = df['SAP2000_id'].astype(int)
# df.to_csv(os.path.join(folder, file), sep=';', index=False)
