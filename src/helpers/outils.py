"""
File Description:
    Contains helpers to be used by main file

Created on 2024/02/06 by asanchezlc
"""
import ast
import copy
from dotenv import load_dotenv
import h5py
import json
import os
import psutil
import pyuff
import re
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
import scipy as sp

import sap2000 as sap2000


def get_username():
    """
    Get the username from the .env file
    """
    load_dotenv()  # loads variables from .env file
    username = os.getenv('USERNAME')
    return username


def get_paths(csv):
    """
    Function duties:
        Get dictionary with paths
    """

    paths_df = pd.read_csv(csv, sep=';', comment='#')
    paths = dict()

    for var in list(paths_df.VarName):
        i = list(paths_df.VarName).index(var)
        path_value = paths_df.Path[i]
        if ":" not in path_value:
            path_value = os.path.join(
                paths_df[paths_df['VarName'] == 'project']['Path'].iloc[0], path_value)
        paths[var] = path_value

    return paths


def increment_filename(filename):
    """
    Function Duties:
        Increment a filename if it already exists in the folder.
    Example
    """
    base, ext = os.path.splitext(filename)
    match = re.search(r'(\d+)$', base)
    if match:  # If a number is found, increment it
        number = int(match.group(1))
        new_base = base[:match.start()] + str(number + 1)
    else:  # If no number is found, add '1' to the base name
        new_base = base + '1'

    # Generate the new filename
    new_filename = new_base + ext

    return new_filename


def merge_all_dict(all_dict):
    """
    Function duties:
        Merges all dictionaries in all_dict into a single one
    Remark:
        Each dictionary contained in all_dict should have the same
        (perhaps not all) keys;
        IMPORTANT: First dictionary keys are used are reference
    """
    dict_merged = dict()
    dict_names = list(all_dict)
    ref_dictionary = all_dict[dict_names[0]]

    for element in list(ref_dictionary):
        dict_merged[element] = dict()

        for dictionary_name in dict_names:
            if all_dict[dictionary_name] is None:
                dict_merged[element][dictionary_name] = None
            elif element in list(all_dict[dictionary_name]):
                dict_merged[element][dictionary_name] = all_dict[dictionary_name][element]

    return dict_merged


def round_6_sign_digits(number):
    """
    Function duties:
        Rounds a float to 6 significant digits
    """
    formatted_number = "{:.6g}".format(number)
    rounded_number = float(formatted_number)

    return rounded_number


def sort_list_string(list_str):
    """
    Function duties:
        Convert a list of string-numbers into a sorted list
        (it is sorted w.r.t. float(list[i]) index)
    """
    float_list = [float(i) for i in list_str]
    sorted_indices = sorted(range(len(float_list)),
                            key=lambda i: float_list[i])
    sorted_list = [list_str[i] for i in sorted_indices]

    return sorted_list


def get_sectionproperties_material(section_properties, section_material,
                                   material_properties):
    """
    Input:
        section_properties: dictionary containing geometric properties
            associated to each section (from get_sectionproperties function)
        section_material: dictionary containing which material is associated
            to each section (from get_material_I_section)
        material_properties: dictionary containing physical properties
            associated to each material (from get_material_properties function)
    Return:
        Dictionary with all aggregated information
    """

    section_properties_material = dict()
    for section in list(section_properties):
        material = section_material[section]
        mat_props = material_properties[material]
        sect_props = section_properties[section]
        section_properties_material[section] = dict()
        section_properties_material[section]['Geometry'] = sect_props
        section_properties_material[section]['Material'] = mat_props

    return section_properties_material


def get_strainmodeshapes(modal_forces, element_section, section_properties_material):
    """
    Function duties:
    Computes the strain mode shapes as follow.
        At each point we have:
            Forces: P, M2, M3
            Section properties: Area, S22, S33
            Material properties: E
        Navier Formula is applied; for that, it is important to know
            what are the coordinates within the function.
    Remark:
        FOR THIS EXAMPLE, FUNCTION IS SUPPOSED TO BE RECTANGULAR
        STRESSES ARE COMPUTED AT UPPER AND DOWN CENTERED POINTS,
        AS WELL AS LEFT AND RIGHT CENTERED POINTS
    """

    strain_modes = dict()
    for mode_label in list(modal_forces):
        strain_modes[mode_label] = dict()
        for element_label in list(modal_forces[mode_label]):
            # Dictionary initialization
            strain_modes[mode_label][element_label] = dict()

            # Get section name
            element = element_label.replace("Element_", "")
            SectionName = element_section[element]

            # Section Properties
            Area = section_properties_material[SectionName]["Geometry"]["Area"]
            S22 = section_properties_material[SectionName]["Geometry"]["S22"]
            S33 = section_properties_material[SectionName]["Geometry"]["S33"]

            # Material Properties
            E = section_properties_material[SectionName]["Material"]["E"]

            # Forces and coordinates
            P = modal_forces[mode_label][element_label]['P']
            M3 = modal_forces[mode_label][element_label]['M3']
            M2 = modal_forces[mode_label][element_label]['M2']
            x = modal_forces[mode_label][element_label]['x']
            mesh_id = modal_forces[mode_label][element_label]['Mesh_id']

            # Stresses [!!!!! MADE FOR RECTANGULAR SECTION !!!!!!]
            sigma_1_2_right = [P[i]/Area - M2[i]/S22 for i, _ in enumerate(P)]
            sigma_1_2_left = [P[i]/Area + M2[i]/S22 for i, _ in enumerate(P)]
            sigma_1_3_up = [P[i]/Area - M3[i]/S33 for i, _ in enumerate(P)]
            sigma_1_3_down = [P[i]/Area + M3[i]/S33 for i, _ in enumerate(P)]

            # Strains
            epsilon_1_2_right = [i/E for i in sigma_1_2_right]
            epsilon_1_2_left = [i/E for i in sigma_1_2_left]
            epsilon_1_3_up = [i/E for i in sigma_1_3_up]
            epsilon_1_3_down = [i/E for i in sigma_1_3_down]

            # Save results
            strain_modes[mode_label][element_label]["x"] = x
            strain_modes[mode_label][element_label]["Mesh_id"] = mesh_id
            strain_modes[mode_label][element_label]["epsilon_1_2_right"] = epsilon_1_2_right
            strain_modes[mode_label][element_label]["epsilon_1_2_left"] = epsilon_1_2_left
            strain_modes[mode_label][element_label]["epsilon_1_3_up"] = epsilon_1_3_up
            strain_modes[mode_label][element_label]["epsilon_1_3_down"] = epsilon_1_3_down

    return strain_modes


def get_strain_timehistory(modal_forces, element_section, section_properties_material):
    """
    Function duties:
    Computes the strain as follow.
        At each point and time step we have:
            Forces: P, M2, M3
            Section properties: Area, S22, S33
            Material properties: E
        Navier Formula is applied; for that, it is important to know
            what are the coordinates within the function.
    Remark:
        FOR THIS EXAMPLE, FUNCTION IS SUPPOSED TO BE RECTANGULAR
        STRESSES ARE COMPUTED AT UPPER AND DOWN CENTERED POINTS,
        AS WELL AS LEFT AND RIGHT CENTERED POINTS
    """
    strain_timehistory = dict()
    for element_label in list(modal_forces):
        # Dictionary initialization
        strain_timehistory[element_label] = dict()

        # Get section name
        element = element_label.replace("Element_", "")
        SectionName = element_section[element]

        # Section Properties
        Area = section_properties_material[SectionName]["Geometry"]["Area"]
        S22 = section_properties_material[SectionName]["Geometry"]["S22"]
        S33 = section_properties_material[SectionName]["Geometry"]["S33"]

        # Material Properties
        E = section_properties_material[SectionName]["Material"]["E"]

        # Forces and coordinates
        P = np.array(modal_forces[element_label]['P'])
        M3 = np.array(modal_forces[element_label]['M3'])
        M2 = np.array(modal_forces[element_label]['M2'])
        x = modal_forces[element_label]['x']

        # Stresses [!!!!! MADE FOR RECTANGULAR SECTION !!!!!!]
        sigma_1_2_right = P/Area - M2/S22
        sigma_1_2_left = P/Area + M2/S22
        sigma_1_3_up = P/Area - M3/S33
        sigma_1_3_down = P/Area + M3/S33
        
        # sigma_1_2_right = [P[i]/Area - M2[i]/S22 for i, _ in enumerate(P)]
        # sigma_1_2_left = [P[i]/Area + M2[i]/S22 for i, _ in enumerate(P)]
        # sigma_1_3_up = [P[i]/Area - M3[i]/S33 for i, _ in enumerate(P)]
        # sigma_1_3_down = [P[i]/Area + M3[i]/S33 for i, _ in enumerate(P)]

        # Strains
        epsilon_1_2_right = sigma_1_2_right/E
        epsilon_1_2_left = sigma_1_2_left/E
        epsilon_1_3_up = sigma_1_3_up/E
        epsilon_1_3_down = sigma_1_3_down/E
        
        # epsilon_1_2_right = [i/E for i in sigma_1_2_right]
        # epsilon_1_2_left = [i/E for i in sigma_1_2_left]
        # epsilon_1_3_up = [i/E for i in sigma_1_3_up]
        # epsilon_1_3_down = [i/E for i in sigma_1_3_down]

        # Save results
        strain_timehistory[element_label]["x"] = x
        # strain_timehistory[element_label]["Mesh_id"] = mesh_id
        strain_timehistory[element_label]["epsilon_1_2_right"] = epsilon_1_2_right.tolist()
        strain_timehistory[element_label]["epsilon_1_2_left"] = epsilon_1_2_left.tolist()
        strain_timehistory[element_label]["epsilon_1_3_up"] = epsilon_1_3_up.tolist()
        strain_timehistory[element_label]["epsilon_1_3_down"] = epsilon_1_3_down.tolist()

    return strain_timehistory


def MaC(phi_X, phi_A):
    """
    If the input arrays are in the form (n,) (1D arrays) the output is a
    scalar, if the input are in the form (n,m) the output is a (m,m) matrix
    (MAC matrix). [m: number of modes]
    Literature:
        [1] Maia, N. M. M., and J. M. M. Silva.
            "Modal analysis identification techniques." Philosophical
            Transactions of the Royal Society of London. Series A:
            Mathematical, Physical and Engineering Sciences 359.1778
            (2001): 29-40.
    :param phi_X: Mode shape matrix X, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :param phi_A: Mode shape matrix A, shape: ``(n_locations, n_modes)``
        or ``n_locations``.
    :return: MAC matrix. Returns MAC value if both ``phi_X`` and ``phi_A`` are
        one-dimensional arrays.
    """
    if phi_X.ndim == 1:
        phi_X = phi_X[:, np.newaxis]

    if phi_A.ndim == 1:
        phi_A = phi_A[:, np.newaxis]

    if phi_X.ndim > 2 or phi_A.ndim > 2:
        raise Exception(
            f'Mode shape matrices must have 1 or 2 dimensions (phi_X: {phi_X.ndim}, phi_A: {phi_A.ndim})')

    if phi_X.shape[0] != phi_A.shape[0]:
        raise Exception(
            f'Mode shapes must have the same first dimension (phi_X: {phi_X.shape[0]}, phi_A: {phi_A.shape[0]})')

    MAC = np.real(np.abs(np.conj(phi_X).T @ phi_A)**2)
    for i in range(phi_X.shape[1]):
        for j in range(phi_A.shape[1]):
            MAC[i, j] = MAC[i, j] /\
                np.real((np.conj(phi_X[:, i]) @ phi_X[:, i] *
                         np.conj(phi_A[:, j]) @ phi_A[:, j]))

    if MAC.shape == (1, 1):
        MAC = MAC[0, 0]

    return MAC


def get_pointconnectivity(all_points_coord, all_elements_coord_connect):
    """
    Input:
        all_points: list of all points in the model
        all_elements_coord_connect: dictionary with the connectivity of all elements
            (coming from get_frameconnectivity function)
    Output:
        all_points_connect_elements: dictionary with the elements connected to each point.
            The dictionary has the following structure:
            {'PointName': {'Element_0': [elements connected to the point having that point as the first node],
                           'Element_f': [elements connected to the point having that point as the final node]}}
    Remark:
        The output dictionary is useful for averaging values in points where multiple
            elements arrive. There is a final warning (which should never raise). If
            that warning arises -> function for averaging might not work properly.
    """
    all_points_connect_elements = dict()
    for point in list(all_points_coord):
        element_0, element_f = list(), list()
        for element in list(all_elements_coord_connect):
            if all_elements_coord_connect[element]['Point_0']['PointName'] == point:
                element_0.append(f'Element_{element}')
            if all_elements_coord_connect[element]['Point_f']['PointName'] == point:
                element_f.append(f'Element_{element}')
        all_points_connect_elements[point] = {
            'Element_0': element_0, 'Element_f': element_f}

    # fast check
    n_tot = 0
    for point in list(all_points_connect_elements):
        n_tot += len(all_points_connect_elements[point]['Element_0']) + len(
            all_points_connect_elements[point]['Element_f'])
    if n_tot > (len(all_points_coord)-2)*2+2:
        message = 'Elements without point assigned'
        warnings.warn(message, UserWarning)
    elif n_tot < (len(all_points_coord)-2)*2+2:
        message = 'Points without element assigned'
        warnings.warn(message, UserWarning)

    return all_points_connect_elements


def prepare_dict_plotting_SG(Psi_id, strain_modeshapes):
    """
    Function Duties:
        Obtains a dictionary for plotting SG (strain gauges) locations
        in the SAP2000 model
    Input:
        Psi_id: list of strings with the element names with the
        station followed by the location (e.g. ['2.0_up', '6.1_right'])
        strain_modeshapes: dictionary with strain mode shapes
            remark: it has information about coordinates
    Output:
        points_for_plotting: dictionary with the information of points
            in which the strain must be measured.
            Example: {'Element_6': {
                                    'x': [0.0625],
                                    'Mesh_id': ['6.1'],
                                    'location': ['up']}
                                    }
    """
    points_for_plotting = dict()
    locations = [i.split('_')[1] for i in list(Psi_id)]
    elements = [i.split('.')[0] for i in list(Psi_id)]
    elements_stats = [i.split('_')[0] for i in list(Psi_id)]
    for element in elements:
        element_label = f'Element_{element}'
        x_all = strain_modeshapes[list(strain_modeshapes)[
            0]][element_label]['x']
        mesh_id_all = strain_modeshapes[list(strain_modeshapes)[
            0]][element_label]['Mesh_id']

        aux_bool = [element == i for i in elements]
        stats = np.array(elements_stats)[aux_bool]

        id_ = [np.where(np.array(mesh_id_all) == i)[0][0] for i in stats]
        x = list(np.array(x_all)[id_])
        mesh_id = list(np.array(mesh_id_all)[id_])
        loc = list(np.array(locations)[aux_bool])

        points_for_plotting[element_label] = {
            'x': x, 'Mesh_id': mesh_id, 'location': loc}

    return points_for_plotting


def average_forces(all_points_connect_elements, modal_forces):
    """
    Input:
        all_points_connect_elements: dictionary with the elements connected to
            each point (coming from get_pointconnectivity function)
        modal_forces: dictionary with the modal forces (coming from get_modalforces
            function)
    Output:
        modal_forces: dictionary with the modal forces averaged in the repeated points
    Remark:
        Only elements that are in the modal_forces (i.e. in "modeshape_frames" group) are
            considered for averaging
    """
    for point in list(all_points_connect_elements):
        repeated_elements_0 = copy.deepcopy(all_points_connect_elements[point]['Element_0'])
        repeated_elements_f = copy.deepcopy(all_points_connect_elements[point]['Element_f'])
        repeated_elements_0_aux = copy.deepcopy(repeated_elements_0)
        repeated_elements_f_aux = copy.deepcopy(repeated_elements_f)
        # remove elements that are not used for retrieving modal forces
        for element_0 in repeated_elements_0_aux:
            if element_0 not in list(modal_forces[list(modal_forces)[0]]):
                repeated_elements_0.remove(element_0)
        for element_f in repeated_elements_f_aux:
            if element_f not in list(modal_forces[list(modal_forces)[0]]):
                repeated_elements_f.remove(element_f)
        repeated_elements = repeated_elements_0 + repeated_elements_f
        if len(repeated_elements) > 1:
            ref_element = repeated_elements[0]
            for mode_label in list(modal_forces):
                # A) substitute the obtained force in the reference element
                for force in ['P', 'M2', 'M3']:
                    # Obtain the force average
                    force_aux_0 = [modal_forces[mode_label][i][force]
                                   for i in repeated_elements_0]
                    force_aux_f = [modal_forces[mode_label][i][force]
                                   for i in repeated_elements_f]
                    force_0_0 = sum(force[0] for force in force_aux_0)
                    force_f_f = sum(force[-1] for force in force_aux_f)

                    force_aver = (force_0_0 + force_f_f) / \
                        (len(force_aux_0) + len(force_aux_f))

                    # Substitute the force in the reference element
                    new_force = copy.deepcopy(
                        modal_forces[mode_label][ref_element][force])
                    if ref_element in repeated_elements_0:
                        new_force[0] = force_aver
                    else:
                        new_force[-1] = force_aver
                    del modal_forces[mode_label][ref_element][force]
                    modal_forces[mode_label][ref_element][force] = new_force

                # B) Remove any reference to the repeated elements
                for element in repeated_elements[1:]:
                    for id_ in list(modal_forces[mode_label][element]):
                        if element in repeated_elements_0:
                            aux = copy.deepcopy(
                                modal_forces[mode_label][element][id_][1:])
                        else:
                            aux = copy.deepcopy(
                                modal_forces[mode_label][element][id_][:-1])
                        del modal_forces[mode_label][element][id_]
                        modal_forces[mode_label][element][id_] = aux

    return modal_forces


def get_element_mesh_coordinates(all_elements_coord_connect, strain_modeshapes,
                                 round_coordinates=True):
    """
    Input:
        all_elements_coord_connect: dictionary coming from get_pointcoordinates function
        strain_modeshapes: dictionary coming from get_strainmodeshapes function
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
    Output:
        id_coord: dictionary containing the coordinates for each element point
            in which the strain is retrieved
    Remark:
        This function only retrieves elements that are in the strain_modeshapes
    """
    id_coord = dict()
    for element in list(all_elements_coord_connect):
        element_coord = all_elements_coord_connect[element]
        element_label = 'Element_' + element
        first_mode = list(strain_modeshapes)[0]
        # We can retrieve only elements that are in the strain_modeshapes
        if element_label not in list(strain_modeshapes[first_mode]):
            continue
        delta_x = strain_modeshapes[first_mode][element_label]['x']
        element_id = strain_modeshapes[first_mode][element_label]['Mesh_id']
        x0, xf = element_coord['Point_0']['x'], element_coord['Point_f']['x']
        y0, yf = element_coord['Point_0']['y'], element_coord['Point_f']['y']
        z0, zf = element_coord['Point_0']['z'], element_coord['Point_f']['z']
        L = np.sqrt((xf - x0)**2 + (yf - y0)**2 + (zf - z0)**2)
        for i, subelement in enumerate(element_id):
            x = x0 + (xf - x0)*delta_x[i]/L
            y = y0 + (yf - y0)*delta_x[i]/L
            z = z0 + (zf - z0)*delta_x[i]/L
            if round_coordinates:
                x, y, z = round_6_sign_digits(
                    x), round_6_sign_digits(y), round_6_sign_digits(z)
            id_coord[subelement] = {'x': x, 'y': y, 'z': z}

    return id_coord


def build_Phi(disp_modeshapes, active_dofs=['U1', 'U2', 'U3'], num_modes=None):
    """
    Function Duties:
        Get the matrix of mode shapes for the selected DOFs
    Input:
        disp_modeshapes: dictionary with the mode shapes coming from
            sap2000.get_displmodeshapes
        active_dofs: list of strings with the DOFS to be included in Phi
        num_modes: number of modes to be included in Phi
    Return:
        Phi: matrix of mode shapes for the selected DOFs and number of modes
        Phi_id: list of strings with the joint names and DOFs for each
            column of Phi
    """
    if num_modes is None:
        num_modes = len(disp_modeshapes)

    joint_id_sorted = sort_list_string(
        disp_modeshapes[list(disp_modeshapes)[0]]['Joint_id'])
    Phi = np.zeros((len(joint_id_sorted) * len(active_dofs), num_modes))
    Phi_id = list()
    for count, dof in enumerate(active_dofs):
        Phi_id += [i + f'_{dof}' for i in joint_id_sorted]
        for mode in range(num_modes):
            mode_label = list(disp_modeshapes)[mode]
            joint_order = [disp_modeshapes[mode_label]
                           ['Joint_id'].index(item) for item in joint_id_sorted]
            Phi[len(joint_id_sorted)*count:len(joint_id_sorted)*(count+1),
                mode] = [disp_modeshapes[mode_label][dof][i] for i in joint_order]

    return Phi, Phi_id


def build_Psi(strain_modeshapes, active_location=['up'], num_modes=None):
    """
    Function Duties:
        Get the matrix of strain mode shapes for the selected strain locations
    Input:
        strain_modeshapes: dictionary with the mode shapes coming from
            sap2000.get_strainmodeshapes
        active_location: list containing one (or multiple) of the following:
            ['right', 'left', 'up', 'down']
        num_modes: number of modes to be included in Phi
    Return:
        Psi: matrix of mode shapes for the selected DOFs and number of modes
        Psi_id: list of strings with the joint names and DOFs for each
            column of Psi
    Remark:
        Psi dimension is n*m, being:
            n: number of points where strain is measured, which is:
                number of elements * number of stations * len(active_location)
            m: number of modes
        Psi_id is a list of strings with the joint names for each element
            (len(Psi_id) = n)
    """
    if num_modes is None:
        num_modes = len(strain_modeshapes)

    # Ver si le quitamos la etiqueta 'Element_' a los elementos
    element_id = list(strain_modeshapes[list(strain_modeshapes)[0]])
    num_points = 0
    for i in element_id:
        num_points += len(strain_modeshapes[list(
            strain_modeshapes)[0]][i]['Mesh_id'])

    Psi_id = list()
    Psi = np.zeros((num_points, num_modes, len(active_location)))
    for count, eps_type in enumerate(active_location):
        Psi_id_aux = list()
        aux_count = 0
        for element in element_id:
            for mode in range(num_modes):
                mode_label = list(strain_modeshapes)[mode]
                eps_type_label = [i for i in list(
                    strain_modeshapes[mode_label][element]) if i.split('_')[-1] == eps_type][0]
                eps = strain_modeshapes[mode_label][element][eps_type_label]
                Psi[aux_count: aux_count+len(eps), mode, count] = eps
                if mode == 0:
                    Psi_id_aux += [
                        f'{i}_{eps_type}' for i in strain_modeshapes[mode_label][element]['Mesh_id']]
            aux_count += len(eps)
        Psi_id.append(Psi_id_aux)

    # Set Psi in format n*m, and Psi_id as a single list
    Psi_rearranged = np.zeros((num_points*len(active_location), num_modes))
    Psi_id_rearranged = list()
    for count in range(len(active_location)):
        Psi_rearranged[count *
                       num_points: (count+1)*num_points, :] = Psi[:, :, count]
        Psi_id_rearranged += Psi_id[count]

    return Psi_rearranged, Psi_id_rearranged


def Phi_selected_dofs(Psi_raw, all_dofs_id, selected_id):
    """
    Input:
        Psi_raw: np array containing all FEM strainmodes
        Psi_id_raw: list containing all FEM dofs id
        Psi_id_selected_dofs: subset of Psi_id_raw
    """
    Psi = np.zeros((len(selected_id), np.shape(Psi_raw)[1]))
    dofs_id = list()

    for i, sg in enumerate(selected_id):
        j = all_dofs_id.index(sg)
        Psi[i, :] = Psi_raw[j, :]
        dofs_id.append(sg)

    return Psi, dofs_id


def Psi_selected_dofs(Psi_raw, all_dofs_id, selected_id):
    """
    Input:
        Psi_raw: np array containing all FEM strainmodes
        Psi_id_raw: list containing all FEM dofs id
        Psi_id_selected_dofs: subset of Psi_id_raw
    """
    Psi = np.zeros((len(selected_id), np.shape(Psi_raw)[1]))
    dofs_id = list()

    for i, sg in enumerate(selected_id):
        j = all_dofs_id.index(sg)
        Psi[i, :] = Psi_raw[j, :]
        dofs_id.append(sg)

    return Psi, dofs_id


def Phi_delete_slaves(Phi, Phi_id, restraints_dict, not_selectable):
    """
    Function Duties:
        Delete slave nodes from Phi and Phi_id
    Input:
        Phi: matrix of mode shapes (coming from build_Phi)
        Phi_id: list of strings with the joint names and DOFs for each
            column of Phi (coming from build_Phi)
        restraints_dict: dictionary with the restraints for each node
            (coming from sap2000.get_pointrestraints)
        not_selectable: list of strings with the nodes not selectable
            (coming from sap2000.getnames_point_elements with nodenotselectable group)
    Return:
        Phi_upd, Phi_id_upd: updated Phi and Phi_id (without slaves nodes)
    """
    # detect nodes not accesible
    not_accesible = np.zeros(len(Phi_id), dtype=bool)
    for node in not_selectable:
        for i, _ in enumerate(Phi_id):
            num = _.split('_')[0]
            if node == num:
                not_accesible[i] = True

    # detect restraint nodes
    restrained = np.zeros(len(Phi_id), dtype=bool)
    # always like that (SAP2000)
    all_dofs = ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']

    active_dofs = list(set([i.split('_')[1] for i in Phi_id]))
    active_dofs = sorted(active_dofs, key=lambda x: int(x[1:]))

    for i, _ in enumerate(Phi_id):
        dof = _.split('_')[1]
        num = _.split('_')[0]
        if dof in active_dofs:
            if restraints_dict[num][all_dofs.index(dof)]:
                restrained[i] = True

    # Remove slaves
    slaves = not_accesible | restrained
    Phi_upd = Phi[~slaves, :]
    Phi_id_upd = np.array(Phi_id)[~slaves]

    return Phi_upd, Phi_id_upd


def adapt_model_results(model_results, joint_coordinates, channel_coordinates):
    """
    Function Duties:
        Adapt results read from SAP2000 to results experimentally
        obtained (i.e. channels coordinates).
    Input:
        model_results: dictionary coming from SAP2000 analysis
        joint_coordinates: dictionary containing joint coordinates in SAP2000
        channel_coordinates: dictionary containing accelerometer coordinates
    """
    # 1. Retrieve frequencies
    model_frequencies = dict()
    for mode_label in list(model_results):
        model_frequencies[mode_label] = model_results[mode_label]['frequencies']['Frequency']

    # 2. Adapt model_results dict (in terms of joints) to channels
    model_modeshapes = dict()
    for mode in list(model_results):
        model_modeshapes[mode] = dict()
        channels, channels_nb, mode_coord = list(), list(), list()
        joint_id = list(model_results[mode]['disp_modeshapes']['Joint_id'])
        for joint in joint_id:
            coord = joint_coordinates[joint]
            for channel in list(channel_coordinates):
                if {key: channel_coordinates[channel][key] for key in ['x', 'y', 'z']} == coord:
                    joint_id = list(
                        model_results[mode]['disp_modeshapes']['Joint_id'])
                    axis = channel_coordinates[channel]["ax"]
                    mode_axis = model_results[mode]['disp_modeshapes'][axis]
                    channels.append(channel)
                    channels_nb.append(int(channel.replace('ch', '')))
                    mode_coord.append(mode_axis[joint_id.index(joint)])
                    # disp_modeshapes[mode][channel] = mode_axis[joint_id.index(joint)]
        id_sorted = [channels_nb.index(i) for i in sorted(channels_nb)]
        channels = [channels[i] for i in id_sorted]
        mode_coord = [mode_coord[i] for i in id_sorted]
        model_modeshapes[mode]['ch_id'] = channels
        model_modeshapes[mode]['coord'] = mode_coord

    # 3. Merge model data into a single dictionary
    all_dict = {'frequency': model_frequencies,
                'disp_modeshapes': model_modeshapes}

    model_modal_unmatched = merge_all_dict(all_dict)

    return model_modal_unmatched


def normalize_mode(fi):
    """
    Input:
        fi: it can be:
            list: indiv. mode shape
            np.array of dim (n, ): indiv. mode shape
            np.array of dim (n, m):
                n: channels
                m: modes
    Return:
        fi_norm: vector normalized (normalizing = max. value equals to 1)
    """
    return_1d = False
    return_list = False
    if isinstance(fi, list):
        return_list = True
    fi = np.array(fi)
    if fi.ndim == 1:
        fi = fi.reshape(len(fi), 1)
        return_1d = True

    fi_norm = np.zeros(np.shape(fi))
    for n_mode in range(np.shape(fi)[1]):
        fi_mode = fi[:, n_mode]
        id_denominator = np.argmax(np.abs(fi_mode))
        fi_norm[:, n_mode] = fi_mode/fi_mode[id_denominator]

    if return_1d:
        fi_norm = fi_norm.reshape(-1)
    if return_list:
        fi_norm = list(fi_norm)

    return fi_norm


def allocate_exp_clusters(modes_cluster, experimental_modal):
    """
    Input:
        modes_cluster: dictionary containing 3 items ('x', 'y', 'theta'),
            each of them having a list with numbers corresponding to each mode
        experimental_modal: dictionary containing modal information for
            each mode, labelled as: 'Mode_1', 'Mode_2', etc.
    Return:
        cluster_modes_exp: modes in experimental_modal arranged according
        to clusters in modes_cluster
    """
    cluster_modes_exp = dict()
    for cluster in list(modes_cluster):
        cluster_modes_exp[cluster] = dict()
        for mode_label in list(experimental_modal):
            mode_number = int(mode_label.replace('Mode_', ''))
            if mode_number in modes_cluster[cluster]:
                cluster_modes_exp[cluster][mode_label] = experimental_modal[mode_label]

    return cluster_modes_exp


def check_cluster_exp(cluster_modes_exp) -> None:
    """
    Function Duties:
        It checks that cluster modes are sorted w.r.t. frequency
        within each cluster
    """
    for cluster in list(cluster_modes_exp):
        frequencies = [cluster_modes_exp[cluster][key]['frequency']
                       for key in list(cluster_modes_exp[cluster])]
        frequencies_sorted = sorted(frequencies)
        if frequencies != frequencies_sorted:
            raise Exception(
                'Clusterized experimental modes are not properly sorted w.r.t. frequency')


def match_modes_cluster(cluster_modes_exp, model_modal_unmatched):
    """
    Function duties: (own procedure by aslc)
        It matches experimental and model mode shapes based on a pre-defined
        cluster. The procedure consists in:
            1) Grouping model_modal_unmatched modes into clusters defined
            in cluster_modes_exp, based on MAC criteria (mode is grouped
            into the cluster with higher MAC).
            2) Match modes based on frequencies: within a cluster, first model
            modes (sorted by frequencies) are assumed to correspond to first
            experimental modes (sorted by frequencies)
    Input:
        cluster_modes_exp: dictionary containing experimental mode shapes
            grouped into clusters. Within each cluster, MODES MUST BE SORTED w.r.t. freq.
        model_modal_unmatched: dictionary containing model mode shapes (not clusterized).
    Output:
        model_modal: dictionary analogous to model_modal_unmatched but with (perhaps)
            different mode numbering (modes should be now matched)
            Additionally, it contains a comparison between the experimental matched
            mode and the corresponding mode
    """
    # Initialize dictionaries
    warn_dict = dict()
    cluster_modes = dict()
    for cluster in list(cluster_modes_exp):
        cluster_modes[cluster] = dict()

    # Check that clusterized modes are sorted w.r.t. frequency
    check_cluster_exp(cluster_modes_exp)

    # Sort model_modal_unmatched w.r.t. frequencies (if it was not sorted)
    frequencies = [model_modal_unmatched[key]['frequency']
                   for key in list(model_modal_unmatched)]
    id_sorted = [frequencies.index(freq) for freq in sorted(frequencies)]
    modes_sorted = [list(model_modal_unmatched)[i] for i in id_sorted]
    model_modal_unmatched = {
        key: model_modal_unmatched[key] for key in modes_sorted}

    # clusterize modes
    for mode in list(model_modal_unmatched):
        mac_cluster = dict()
        for cluster in list(cluster_modes_exp):
            mac_cluster_list = list()
            for mode_cluster in list(cluster_modes_exp[cluster]):
                mode_exp = cluster_modes_exp[cluster][mode_cluster]['disp_modeshapes']['coord']
                mode_model = model_modal_unmatched[mode]['disp_modeshapes']['coord']
                mac = MaC(np.array(mode_model), np.array(mode_exp))
                mac_cluster_list.append(mac)
            mac_cluster[cluster] = np.max(mac_cluster_list)
        cluster_matched = list(cluster_modes_exp)[np.argmax(
            [mac_cluster[key] for key in list(mac_cluster)])]
        cluster_modes[cluster_matched][mode] = copy.deepcopy(
            model_modal_unmatched[mode])

    # # reallocate modes ---> TO BE FINISHED
    # lacking_cluster_bool = [len(cluster_modes_exp[cluster]) > len(cluster_modes[cluster]) for cluster in list(cluster_modes_exp)]
    # excessive_cluster_bool = [len(cluster_modes_exp[cluster]) < len(cluster_modes[cluster]) for cluster in list(cluster_modes_exp)]
    # lacking_cluster = [list(cluster_modes_exp)[i] for i, bool in enumerate(lacking_cluster_bool) if bool]
    # excessive_cluster = [list(cluster_modes_exp)[i] for i, bool in enumerate(excessive_cluster_bool) if bool]
    # for cluster_exc in excessive_cluster:
    #     for mode in list(cluster_modes[cluster_exc]):
    #         for cluster_lack in lacking_cluster:
    #             # we compare mode in excessive with all experimental modes in lacking
    #             mode_exp = cluster_modes_exp[cluster][mode_cluster]['disp_modeshapes']['coord']
    #             mode_model = model_modal_unmatched[mode]['disp_modeshapes']['coord']
    #             mac = MaC(np.array(mode_model), np.array(mode_exp))
    #             mac_cluster_list.append(mac)

    # frequency-based modes matching
    model_modal = dict()
    for cluster in list(cluster_modes):
        matched_modes = list(cluster_modes_exp[cluster])
        if len(cluster_modes_exp[cluster]) > len(cluster_modes[cluster]):
            message = f'number of experimental modes > number of model modes in cluster {cluster}'
            matched_modes = matched_modes[0:len(cluster_modes[cluster])]
            warn_dict[cluster] = message

        for jj, mode in enumerate(matched_modes):
            # Assignment based on frequency order
            mode_model_name = list(cluster_modes[cluster])[jj]
            model_modal[mode] = copy.deepcopy(
                model_modal_unmatched[mode_model_name])

            # Frequency comparison
            delta_freq = np.abs(
                cluster_modes_exp[cluster][mode]['frequency'] - model_modal[mode]['frequency'])

            # MAC comparison
            mode_exp = cluster_modes_exp[cluster][mode]['disp_modeshapes']['coord']
            mode_model = model_modal[mode]['disp_modeshapes']['coord']
            mac = MaC(np.array(mode_model), np.array(mode_exp))

            # Add comparison
            comparison = {'delta_f': delta_freq, 'MAC': mac}
            model_modal[mode]['comparison'] = comparison

    # sort dictionary
    model_modal = {key: model_modal[key] for key in sorted(list(model_modal))}

    return model_modal, warn_dict


def match_modes(model_modal_unmatched, experimental_modal, mac_threshold=0.0):
    """
    Function Duties:
        Match modes obtained in the model with experimental modes
    Working principle: (to be checked by Enrique or Juan)
        Each pair {freq, mode} from the model is compared against all experimental
        information {freq_exp_all, modes_exp_all};
        All modes within range [max_mac - mac_threshold, max_mac] are retrieved;
        nearest mode (in terms of frequency) is retrieved
        If more than one model modes are assigned to one experimental mode, a
        comparison is made with same procedure, retrieving the mode with max(mac)
        (except if mac values are closer than mac_threshold, when mode with closest
        freq is retrieved)
    Input:
        model_modal_unmatched: modal results coming from SAP2000 (with coordinates
            adapted to channel format, i.e., using adapt_model_results function)
        experimental_modal: experimental mode shapes
        mac_threshold: if 0 -> mode with higher mac is retrieved; if smaller, closer
            frequencies get more importance [set >0 only if frequencies are well adjusted]
    Output:
        model_modal: dictionary analogous to model_modal_unmatched but with (perhaps)
            different mode numbering (modes should be now matched)
    """
    # Necessary variables
    model_modal = dict()
    freq_exp_all = [experimental_modal[mode]['frequency']
                    for mode in list(experimental_modal)]
    modes_exp_all = np.array([experimental_modal[mode]['disp_modeshapes']['coord']
                             for mode in list(experimental_modal)])
    modes_exp_all = modes_exp_all.T
    ch_exp = experimental_modal['Mode_1']['disp_modeshapes']['ch_id']
    count_assigned = 0

    # Check that num_modes_model >= num_modes_experimental
    if len(experimental_modal) > len(model_modal_unmatched):
        message = 'The number of experimental modes is larger than that of the model'
        warnings.warn(message, UserWarning)

    # Mode matching (multiple modes can be assigned to same one)
    for mode_label in list(model_modal_unmatched):
        # Frequency comparison
        freq_model = model_modal_unmatched[mode_label]['frequency']
        delta_freq = np.array([np.abs(freq_model - f_exp)
                              for f_exp in freq_exp_all])

        # MAC comparison
        mode_model_aux = model_modal_unmatched[mode_label]['disp_modeshapes']['coord']
        ch_model = model_modal_unmatched[mode_label]['disp_modeshapes']['ch_id']
        mode_model = [mode_model_aux[ch_model.index(i)] for i in ch_exp]
        mac = MaC(np.array(mode_model), np.array(modes_exp_all))[0]
        delta_mac = np.array([np.max(mac)-i for i in mac])

        # Mode matching (minimum frequency difference in values with maximum mac)
        mac_mask = delta_mac <= mac_threshold
        id_mode = np.where(delta_freq == np.min(delta_freq[mac_mask]))[0][0]
        mode_assigned = list(experimental_modal)[id_mode]
        # if mode had already been assigned
        if mode_assigned in list(model_modal):
            count_assigned += 1
            mode_assigned += f'_R_{count_assigned}'
        model_modal[mode_assigned] = copy.deepcopy(
            model_modal_unmatched[mode_label])
        comparison = {'delta_f': delta_freq[id_mode], 'MAC': mac[id_mode]}
        model_modal[mode_assigned]['comparison'] = comparison

    # Remove extra assigned modes
    extra_modes = [key for key in list(model_modal) if 'R' in key]
    for extra_key in extra_modes:
        base_mode = extra_key.split('_R')[0]
        # We respect former procedure (compare mac and, if below threshold, smaller delta_f)
        mac = np.array([model_modal[base_mode]['comparison']
                       ['MAC'], model_modal[extra_key]['comparison']['MAC']])
        delta_freq = np.array([model_modal[base_mode]['comparison']
                              ['delta_f'], model_modal[extra_key]['comparison']['delta_f']])
        delta_mac = np.array([np.max(mac)-i for i in mac])
        mac_mask = delta_mac < mac_threshold
        id_mode = np.where(delta_freq == np.min(delta_freq[mac_mask]))[0][0]

        if id_mode == 0:
            del model_modal[extra_key]
        else:
            del model_modal[base_mode]
            model_modal[base_mode] = model_modal.pop(extra_key)

    # Sort dictionary
    model_modal = {key: model_modal[key] for key in sorted(list(model_modal))}

    return model_modal


def from_input_to_parameters(input_data, sigma, dispersion):
    """
    Function Duties:
        Gets input_data dictionary (defined for SAP2000) and returns
        Parameters dictionary (suitable for Bayesian Inference functions)
    Input:
        input_data: dictionary to be used to feed SAP2000 model
        sigma: variance of the data (necessary to define stochastic model,
            for the normal distribution component)
        dispersion: dispersion of the data (necessary to define stochastic
            model, for the log-normal distribution component)
    Output:
        Parameters: dictionary to be used to feed bayesian functions
    Remark:
        THIS FUNCTION IS NOT FURTHER USED
    """
    Parameters = dict()
    parameters_groups = list(input_data)
    for group in parameters_groups:
        if group == 'material':
            for i in ['E', 'rho']:
                # only one material is defined
                material = list(input_data[group])[0]
                Parameters[i] = input_data[group][material][i]
        elif group == 'joint_springs':
            # only one joint group is defined
            joint_group = list(input_data[group])[0]
            displacement_labels = ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']
            for id in [3, 4]:  # 3, 4: R1 and R2
                Parameters[f'spring_{displacement_labels[id]}'] = input_data[group][joint_group][id]
        elif group == 'frame_releases':
            releases_labels = ['U1', 'U2', 'U3', 'T', 'M2', 'M3']
            for frame_group in list(input_data[group]):
                for j in [3, 4, 5]:  # 3, 4, 5: T, M2 and M3 // only EndValue is retrieved
                    Parameters[frame_group + '_' + releases_labels[j]
                               ] = input_data[group][frame_group]['EndValue'][j]

    Parameters['sigma'] = sigma
    Parameters['dispersion'] = dispersion

    return Parameters


def from_parameters_to_input(parameters_array, material_name='frame-Steel',
                             joint_name='supportpoints', frame_name='releaseframes',
                             parameters_groups=['material', 'joint_springs', 'frame_releases']):
    """
    Function Duties:
        Gets Parameters dictionary (defined for SAP2000) and returns
        input_data dictionary (suitable for Bayesian Inference functions)
    Input:
        Parameters: dictionary to be used to feed bayesian functions
    Output:
        input_data: dictionary to be used to feed SAP2000 model
    """
    # 1. Transform ndarray into numbers, lists, etc.
    parameters = dict()
    for i, key in enumerate(parameters_array):
        if isinstance(parameters_array[key], np.ndarray):
            if parameters_array[key].ndim == 1 and len(parameters_array[key]) == 1:
                parameters[key] = parameters_array[key][0]
            else:
                parameters[key] = parameters_array[key].tolist()
        else:
            parameters[key] = parameters_array[key]

    # 2. Build input_data dictionary
    input_data = dict()
    for group in parameters_groups:
        input_data[group] = dict()

        # 2.1. Adapt "material" category
        if group == 'material':
            input_data[group][material_name] = dict()
            for _, key in enumerate(['E', 'rho']):
                input_data[group][material_name][key] = parameters[key]

        # 2.1. Adapt "joint_springs" category (to be used in the future)
        elif group == 'joint_springs':
            displacement_labels = ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']
            joint_mask = [string.startswith(joint_name) for string in list(parameters)]
            group_numbers = [string.split('_')[1] for i, string in enumerate(
                list(parameters)) if joint_mask[i]]
            if len(set(group_numbers)) > 1:
                group_numbers = sort_list_string(list(set(group_numbers)))
            for number in group_numbers:
                input_data[group][f'{joint_name}_{number}'] = dict()
                joint_values = [0 for i in range(len(displacement_labels))]
                for i, key in enumerate(displacement_labels):
                    parameters_key = f'{joint_name}_{number}_{key}'
                    if parameters_key in list(parameters):
                        joint_values[i] = parameters[parameters_key]
                input_data[group][f'{joint_name}_{number}'] = [i for i in joint_values]

        # 2.3. Adapt "frame_releases" category
        elif group == 'frame_releases':
            """
            ii and StartValue are set to 0 and False lists
            EndValue is taken from parameters
            jj values linked to non-zero EndValues are set to True
            """
            releases_labels = ['U1', 'U2', 'U3', 'T', 'M2', 'M3']
            frame_mask = [string.startswith(frame_name)
                          for string in list(parameters)]
            group_numbers = [string.split('_')[1] for i, string in enumerate(
                list(parameters)) if frame_mask[i]]
            if len(set(group_numbers)) > 1:  # all not assumed to be inside (reorganize properly otherwise)
                group_numbers = sort_list_string(list(set(group_numbers)))
            for number in group_numbers:
                input_data[group][f'{frame_name}_{number}'] = dict()
                frame_values_end = [0 for i in range(len(releases_labels))]
                for i, key in enumerate(releases_labels):
                    parameters_key = f'{frame_name}_{number}_{key}'
                    if parameters_key in list(parameters):
                        frame_values_end[i] = parameters[parameters_key]
                frame_values_ii = [False for i in range(
                    len(releases_labels))]  # always false
                frame_values_start = [0 for i in range(
                    len(releases_labels))]  # always 0
                frame_values_jj = [frame_values_end[i] !=
                                   0 for i in range(len(releases_labels))]
                input_data[group][f'{frame_name}_{number}']['ii'] = [
                    i for i in frame_values_ii]
                input_data[group][f'{frame_name}_{number}']['StartValue'] = [
                    i for i in frame_values_start]
                input_data[group][f'{frame_name}_{number}']['jj'] = [
                    i for i in frame_values_jj]
                input_data[group][f'{frame_name}_{number}']['EndValue'] = [
                    i for i in frame_values_end]

    return input_data


def EFI(Phi, Phi_id, n_sensors):
    """
    Function Duties:
        Removes the least informative sensor from the set of
        sensors through the EFI algorithm
    Input:
        Phi: Matrix of mode shapes
        Phi_id: List of joint names
        n_sensors: Number of sensors to be maintained
    References:
        Enrique García-Macías' code-based implementation
        Kammer, D.C. Sensor placement for on-orbit modal identification
        and correlation of large space structures. J. Guid. Control Dyn.
        1991, 14, 251–259.
    """
    n_iter = len(Phi_id) - n_sensors
    detFIM, Ed_all = np.zeros(n_iter), list()
    deleted_channels = list()
    for i in range(n_iter):
        Q = np.dot(Phi.T, Phi)
        detFIM[i] = np.linalg.det(Q)
        Lambda, Psi = np.linalg.eig(Q)
        G = np.multiply(np.dot(Phi, Psi), np.dot(Phi, Psi))
        Fe = np.dot(G, np.linalg.inv(np.diag(Lambda)))
        Ed = np.sum(Fe, axis=1)
        # Ed = [np.dot(np.dot(np.dot(Phi, Psi), np.linalg.inv(np.diag(Lambda))), np.dot(Psi.T, Phi.T))[i, i] for i in range(np.shape(Phi)[0])]
        Ed_all.append(Ed.tolist())
        ranked_sensors_id = np.argsort(Ed)

        if Ed[ranked_sensors_id[0]] > 0.99:
            message = 'All sensors are informative'
            warnings.warn(message, UserWarning)

        channel_deleted_id = ranked_sensors_id[0]
        channel_deleted = Phi_id[channel_deleted_id]
        deleted_channels.append(channel_deleted)

        Phi = np.delete(Phi, channel_deleted_id, axis=0)
        Phi_id = np.delete(Phi_id, channel_deleted_id)

    EFI_results = dict()
    EFI_results['results'] = {'Phi': Phi.tolist(), 'Phi_id': Phi_id.tolist()}
    EFI_results['process'] = {'detFIM': detFIM.tolist(), 'Ed_all': Ed_all,
                              'deleted_channels': deleted_channels}

    return EFI_results


def complex_to_normal_mode(mode, max_dof=50, long=True):
    """Transform a complex mode shape to normal mode shape.
    [From EGM codes]

    The real mode shape should have the maximum correlation with
    the original complex mode shape. The vector that is most correlated
    with the complex mode, is the real part of the complex mode when it is
    rotated so that the norm of its real part is maximized. [1]
    ``max_dof`` and ``long`` arguments are given for modes that have
    a large number of degrees of freedom. See ``_large_normal_mode_approx()``
    for more details.

    Literature:
        [1] Gladwell, H. Ahmadian GML, and F. Ismail.
            "Extracting Real Modes from Complex Measured Modes."
            (avaliable in 'doc' folder)

    :param mode: np.ndarray, a mode shape to be transformed. Can contain a single
        mode shape or a modal matrix `(n_locations, n_modes)`.
    :param max_dof: int, maximum number of degrees of freedom that can be in
        a mode shape. If larger, ``_large_normal_mode_approx()`` function
        is called. Defaults to 50.
    :param long: bool, If True, the start in stepping itartion is altered, the
        angles of rotation are averaged (more in ``_large_normal_mode_approx()``).
        This is needed only when ``max_dof`` is exceeded. The normal modes are
        more closely related to the ones computed with an entire matrix. Defaults to True.
    :return: normal mode shape
    """
    if mode.ndim == 1:
        mode = mode[None, :, None]
    elif mode.ndim == 2:
        mode = mode.T[:, :, None]
    else:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim}).')

    # if mode.shape[1] > max_dof   --> Computationally expensive
    if mode.shape[1] > max_dof:
        return _large_normal_mode_approx(mode[:, :, 0].T, step=int(np.ceil(mode.shape[1] / max_dof)) + 1, long=long)

    # 1. Normalize modes so that norm == 1.0
    _norm = np.linalg.norm(mode, axis=1)[:, None, :]
    mode = mode / _norm

    # 2. Obtain U matrix
    mode_T = np.transpose(mode, [0, 2, 1])
    U = np.matmul(np.real(mode), np.real(mode_T)) + \
        np.matmul(np.imag(mode), np.imag(mode_T))

    # Modification to operate without nan values (otherwise np.linalg.eig raise error)
    nan_mode = np.all(np.isnan(U), axis=(1, 2))
    nan_index = np.where(nan_mode)[0]
    if nan_index.size > 0:
        not_nan = [not (i) for i in nan_mode]
        U_copy = U[not_nan, :, :]
    else:
        U_copy = U

    # 3. Obtain eigenvectors & eigenvalues and choose eigenvector associated to max eigenvalue
    val, vec = np.linalg.eig(U_copy)
    # modification to get as a result mode=0 for nan values [spureous modes]
    if nan_index.size > 0:
        val_aux = np.empty((np.shape(val)[0]+len(nan_index), np.shape(val)[1]))
        vec_aux = np.empty(
            (np.shape(U_copy)[0]+len(nan_index), np.shape(U_copy)[1], np.shape(U_copy)[2]))
        val_aux[not_nan, :] = val
        vec_aux[not_nan, :, :] = vec
        for j in nan_index:
            val_aux[not_nan, :] = np.zeros((np.shape(U_copy)[1]))
            vec_aux[j, :, :] = np.zeros(
                (np.shape(U_copy)[1], np.shape(U_copy)[2]))
        i = np.argmax(np.real(val_aux), axis=1)
        normal_mode = np.real([v[:, _] for v, _ in zip(vec_aux, i)]).T
    else:  # in normal cases we are here
        i = np.argmax(np.real(val), axis=1)
        normal_mode = np.real([v[:, _] for v, _ in zip(vec, i)]).T

    return normal_mode


def _large_normal_mode_approx(mode, step, long):
    """Get normal mode approximation for large modes.
    [From EGM codes]

    In cases, where ``mode`` has ``n`` coordinates and
    ``n`` is large, this would result in a matrix ``U`` of
    size ``n x n``. To find eigenvalues of this non-sparse
    matrix is computationally expensive. The solution is to
    find the angle of the rotation for the vector - this is
    done using only every ``step`` element of ``mode``.
    The entire ``mode`` is then rotated, thus the full normal
    mode is obtained.

    Síntesis: buscar la orientación media de las componentes
    modales en el plano complejo para, a posteriori, rotar las
    componentes y tratar de alinearlas en el eje rea

    To ensure the influence of all the coordinates, a ``long``
    parameter can be used. Multiple angles of rotation are
    computed and then averaged.

    :param mode: a 2D mode shape or modal matrix ``(n_locations x n_modes)``
    :param step: int, every ``step`` elemenf of ``mode`` will be taken
        into account for angle of rotation calculation.
    :param long: bool, if True, the angle of rotation is computed
        iteratively for different starting positions (from 0 to ``step``), when
        every ``step`` element is taken into account.
    :return: normal mode or modal matrix of ``mode``.
    """
    if mode.ndim == 1:
        mode = mode[:, None]
    elif mode.ndim > 2:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim})')

    mode = mode / np.linalg.norm(mode, axis=0)[None, :]

    if long:
        step_long = step
    else:
        step_long = 1

    Alpha = []
    for i in range(step_long):
        mode_step = mode[i::step]
        mode_normal_step = complex_to_normal_mode(mode_step)

        v1 = np.concatenate(
            (np.real(mode_step)[:, :, None], np.imag(mode_step)[:, :, None]), axis=2)
        v2 = np.concatenate((np.real(mode_normal_step)[:, :, None], np.imag(
            mode_normal_step)[:, :, None]), axis=2)

        v1 /= np.linalg.norm(v1, axis=2)[:, :, None]
        v2 /= np.linalg.norm(v2, axis=2)[:, :, None]

        dot_product = np.array([np.matmul(np.transpose(v1[:, j, :, None], [0, 2, 1]),
                               v2[:, j, :, None]) for j in range(v1.shape[1])])
        angles = np.arccos(dot_product)

        alpha = np.mean(angles[:, :, 0, 0], axis=1)
        Alpha.append(alpha)

    alpha = np.mean(Alpha, axis=0)[None, :]

    mode_normal_full = np.real(mode)*np.cos(alpha) - \
        np.imag(mode)*np.sin(alpha)
    mode_normal_full /= np.linalg.norm(mode_normal_full, axis=0)[None, :]

    return mode_normal_full


def get_PSD_SVD(data, window='hann', pov=0.5, df=0.01):
    """
    Input:
        data: df with time and channels
            REMARK: time must be called 't'
        window: window type
        pov: overlap percentage
        df: desired frequency resolution
    Output:
        f, PSD
    """
    t = list(data['t'])
    channels = list(data.drop(columns=['t']))
    sps = len(t)/(t[-1]-t[0])
    L = len(t)
    nxseg = int(sps/df)

    if nxseg > L:
        df_old = float(df)
        nxseg = L
        df = sps/nxseg
        message1 = f'desired frequency resolution df={"{:.2g}".format(df_old)}Hz is not achievable;'
        message2 = f'setting df={"{:.2g}".format(df)}Hz instead to achieve maximum df possible;'
        warnings.warn(message1 + ' ' + message2, UserWarning)

    noverlap = pov * nxseg  # number of overlapping points
    num_windows = int(np.floor(L-noverlap)/(nxseg-noverlap)
                      )  # total number of windows

    PSD_matr = np.zeros((len(channels), len(channels),
                        int((nxseg)/2+1)), dtype=complex)

    # PSD
    for i, ch_i in enumerate(channels):
        for j, ch_j in enumerate(channels):
            f, Pxy = sp.signal.csd(data[ch_i], data[ch_j], fs=sps, nperseg=nxseg,
                                   noverlap=noverlap, window=window)
            PSD_matr[i, j, :] = Pxy

    S_val = np.zeros((len(PSD_matr), len(PSD_matr), int((nxseg)/2+1)))
    for i in range(np.shape(PSD_matr)[2]):
        U1, S1, _V1_t = np.linalg.svd(PSD_matr[:, :, i])
        U1_1 = np.transpose(U1)
        S1 = np.diag(S1)
        S_val[:, :, i] = S1

    return (f, PSD_matr), (S_val, U1_1)


def extract_info_from_name(test_name: str) -> dict:
    """
    Function Duties:
        - Extract information from the test name
    """
    information = dict()
    # Check for standalone numbers
    match = re.search(r'\d+', test_name)
    if match:
        information['test_number'] = int(match.group())
    else:
        raise ValueError('No test number found in the test name')

    # Assign data type
    if 'acc' in test_name:
        information['data_type'] = 'Acceleration'
    else:
        information['data_type'] = 'Strain'

    # Check for 'Nsg'
    match_sg = re.search(r'(\d+)sg', test_name)
    if match_sg:
        information['num_channels'] = int(match_sg.group(1))
    else:
        information['num_channels'] = None

    # Check for 'suboptN'
    match_subopt = re.search(r'subopt(\d+)', test_name)
    if match_subopt:
        information['num_subopt'] = int(match_subopt.group(1))
    else:
        information['num_subopt'] = None

    return information


def uff_read_nodes_modeshapes(filepath):
    """
    Function Duties:
        Reads data from uff file coming from ARTeMIS
    Remark:
        That file is generated by estimating mode shapes in artemis, selecting
        them and exporting them to a .uff file
    """
    uff = pyuff.UFF(filepath)

    # Read all data sets in the UFF file
    data_sets = uff.read_sets()

    # Initialize containers for nodes, trace lines, surfaces, and mode shapes
    nodes = []
    trace_lines = []
    surfaces = []
    mode_shapes = []

    # Iterate over each data set and process based on the type
    for data_set in data_sets:
        if data_set['type'] == 15:
            nodes.append(data_set)
        elif data_set['type'] == 82:
            trace_lines.append(data_set)
        elif data_set['type'] == 2412:
            surfaces.append(data_set)
        elif data_set['type'] == 55:
            mode_shapes.append(data_set)

    return nodes, mode_shapes


def get_channels_suboptimal(test_properties, subopt):
    """
    Function Duties:
        Gets the channels used in the current test, given the test_properties and the
        suboptimal configuration
    Input:
        test_properties: Dictionary with the test properties
        subopt: DataFrame with the coordinates of the chosen suboptimal configurations
    """
    channels_num = subopt['Channels'][subopt['Number']
                                      == test_properties['num_subopt']]
    channels_num = channels_num.to_string(index=False)
    s = channels_num.strip('[]')
    numbers = re.split(r',\s*', s)
    channels = [
        f'channel_{int(num)}' for num in sort_list_string(numbers)]

    return channels


def get_channels_optimal_N_channels(sg_efi_file, test_info):
    """
    Function Duties:
        Gets the channels used in the current test, given the test_info (with all measured
        channels) and the sg_efi_file (which contains channels coordinates without channel number)
    Input:
        sg_efi_file: File generated by EfI.py file
        test_info: File generated by From_Arduino_2_Artemis.py file
    """
    channels = list()
    for ch in test_info['coordinates']:
        exists = False
        for j in range(len(sg_efi_file)):
            coords = {
                'x': sg_efi_file['x'][j], 'y': sg_efi_file['y'][j], 'z': sg_efi_file['z'][j]}
            if coords == test_info['coordinates'][ch]:
                exists = True
                break
        if exists:
            channels.append(ch)
    return channels


def build_Phi_ARTeMIS(selected_channels, nodes_artemis, mode_shapes_artemis):
    """
    Input:
        selected_channels: dictionary containing information about the channels that
            are considered in the oma
        nodes_artemis: all nodes from the artemis model
        mode_shapes_artemis: mode shapes from the artemis model
    Output:
        frequencies_artemis: dictionary containing the frequencies of the mode shapes
        Phi_artemis: matrix containing the mode shapes
        Phi_artemis_id: list containing the id of the channels
    """
    frequencies_artemis = dict()
    Phi_artemis = np.zeros((len(selected_channels), len(
        mode_shapes_artemis)), dtype=complex)
    Phi_artemis_id = list()

    for i, ch in enumerate(selected_channels):
        coord = selected_channels[ch]['coordinates']
        for j, node in enumerate(nodes_artemis[0]['node_nums']):
            x = nodes_artemis[0]['x'][j]
            y = nodes_artemis[0]['y'][j]
            z = nodes_artemis[0]['z'][j]
            direction = selected_channels[ch]["directions"] if selected_channels[ch][
                "directions"][0] != '-' else selected_channels[ch]["directions"][1:]
            if coord['x'] == x and coord['y'] == y and coord['z'] == z:
                for k, _ in enumerate(mode_shapes_artemis):
                    x_mode = mode_shapes_artemis[k]['r1'][j]
                    y_mode = mode_shapes_artemis[k]['r2'][j]
                    z_mode = mode_shapes_artemis[k]['r3'][j]
                    if direction == 'U1':
                        Phi_artemis[i, k] = x_mode
                    elif direction == 'U2':
                        Phi_artemis[i, k] = y_mode
                    elif direction == 'U3':
                        Phi_artemis[i, k] = z_mode
                    else:
                        message = f'Direction must be either U1, U2, U3, but {direction} was given'
                        warnings.warn(message, UserWarning)
                    if i == 0:
                        match = re.search(
                            r'f=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)Hz', mode_shapes_artemis[k]['id3'])
                        frequencies_artemis['Mode_' +
                                            str(k+1)] = float(match.group(1))
                Phi_artemis_id.append(
                    f'{str(selected_channels[ch]["id"])}_{direction}')

    return frequencies_artemis, Phi_artemis, Phi_artemis_id


def build_Psi_timehistory(strain_timehistory, active_location=['up'], num_time_steps=None):
    """
    Function Duties:
        Get the matrix of time history strains
    Input:
        strain_timehistory: dictionary with the strain time history
            sap2000.get_strain_timehistory
        active_location: list containing one (or multiple) of the following:
            ['right', 'left', 'up', 'down']
        num_time_steps: number of time steps in the time history
    Return:
        Psi: matrix containing strains
        Psi_id: list of strings with the joint names and DOFs for each
            column of Psi
    Remark:
        Psi dimension is n*m, being:
            n: number of points where strain is measured, which is:
                number of elements * number of stations * len(active_location)
            m: number of time steps
        Psi_id is a list of strings with the joint names for each element
            (len(Psi_id) = n)
    """
    if num_time_steps is None:
        aux = strain_timehistory[list(strain_timehistory)[0]]
        for i in list(aux):
            if 'epsilon' in i:
                num_time_steps = len(aux[i][0])
                break

    element_id = list(strain_timehistory)
    num_points = 0
    for i in element_id:
        num_points += len(strain_timehistory[i]['x'])

    Psi_id = list()
    Psi = np.zeros((num_points, num_time_steps, len(active_location)))
    for count, eps_type in enumerate(active_location):
        Psi_id_aux = list()
        aux_count = 0
        for j, element in enumerate(element_id):
            eps_type_label = [i for i in list(
                strain_timehistory[element]) if i.split('_')[-1] == eps_type][0]
            eps = strain_timehistory[element][eps_type_label]
            Psi[aux_count: aux_count+len(eps), :, count] = eps
            mesh = [f'{element}.{mesh_id}' for mesh_id in range(len(strain_timehistory[element]['x']))]
            Psi_id_aux += [f'{i}_{eps_type}' for i in mesh]
            aux_count += len(eps)
        Psi_id.append(Psi_id_aux)

    # Set Psi in format n*m, and Psi_id as a single list
    Psi_rearranged = np.zeros((num_points*len(active_location), num_time_steps))
    Psi_id_rearranged = list()
    for count in range(len(active_location)):
        Psi_rearranged[count * num_points: (count+1)*num_points, :] = Psi[:, :, count]
        Psi_id_rearranged += Psi_id[count]

    return Psi_rearranged, Psi_id_rearranged


def build_Psi_ARTeMIS(selected_channels, nodes_artemis, mode_shapes_artemis,
                      tolerance=None):
    """
    Input:
        selected_channels: dictionary containing information about the channels that
            are considered in the oma
        nodes_artemis: all nodes from the artemis model
        mode_shapes_artemis: mode shapes from the artemis model
        tolerance: if value is specified, points are accepted if distance
            is lower to tolerance
    Output:
        frequencies_artemis: dictionary containing the frequencies of the mode shapes
        Psi_artemis: matrix containing the mode shapes
        Psi_artemis_id: list containing the id of the channels
    """
    frequencies_artemis = dict()
    Psi_artemis = np.zeros((len(selected_channels), len(
        mode_shapes_artemis)), dtype=complex)
    Psi_artemis_id = list()

    for i, ch in enumerate(selected_channels):
        coord = selected_channels[ch]['coordinates']
        for j, node in enumerate(nodes_artemis[0]['node_nums']):
            x = nodes_artemis[0]['x'][j]
            y = nodes_artemis[0]['y'][j]
            z = nodes_artemis[0]['z'][j]
            is_point = False
            if tolerance is not None:
                distance = np.sqrt((coord['x']-x)**2 + (coord['y']-y)**2 + (coord['z']-z)**2)
                if distance < tolerance:
                    is_point = True
            else:
                is_point = coord['x'] == x and coord['y'] == y and coord['z'] == z
            if is_point:
                for k, _ in enumerate(mode_shapes_artemis):
                    x_mode = mode_shapes_artemis[k]['r1'][j]
                    y_mode = mode_shapes_artemis[k]['r2'][j]
                    z_mode = mode_shapes_artemis[k]['r3'][j]
                    direction = selected_channels[ch]['directions']
                    if direction == 'x':
                        Psi_artemis[i, k] = x_mode
                        if y_mode != 0 or z_mode != 0:
                            message = f'Mode shape for channel {ch} is not only in the x direction'
                            warnings.warn(message, UserWarning)
                            Psi_artemis[i, k] = np.sqrt(x_mode**2 + y_mode**2 + z_mode**2)
                    elif direction == 'y':
                        Psi_artemis[i, k] = y_mode
                        if x_mode != 0 or z_mode != 0:
                            message = f'Mode shape for channel {ch} is not only in the y direction'
                            warnings.warn(message, UserWarning)
                            Psi_artemis[i, k] = np.sqrt(x_mode**2 + y_mode**2 + z_mode**2)
                    elif direction == 'z':
                        Psi_artemis[i, k] = z_mode
                        if x_mode != 0 or y_mode != 0:
                            message = f'Mode shape for channel {ch} is not only in the z direction'
                            warnings.warn(message, UserWarning)
                            Psi_artemis[i, k] = np.sqrt(x_mode**2 + y_mode**2 + z_mode**2)
                    if i == 0:
                        match = re.search(
                            r'f=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)Hz', mode_shapes_artemis[k]['id3'])
                        frequencies_artemis['Mode_' +
                                            str(k+1)] = float(match.group(1))
                Psi_artemis_id.append(
                    f'{str(selected_channels[ch]["id"])}_{selected_channels[ch]["dir"]}')

    return frequencies_artemis, Psi_artemis, Psi_artemis_id


def get_joint_matrix_index(FilePath):
    """
    Function Duties:
        Read the joint matrix index from a file
    Input:
        FilePath: Path to the .TXE file
    Output:
        joints_matrix_index: DataFrame containing the joint matrix index
    REMARK:
        Joint index ARE ADJUSTED to Python indexing (start at 0)
    """
    # Read the file
    with open(FilePath, 'r') as file:
        lines = file.readlines()

    # Process the lines (skipping the header line)
    processed_lines = []
    temp_line = ""
    header_skipped = False
    for line in lines:
        stripped_line = line.strip()
        if not header_skipped:
            header_skipped = True
            continue  # Skip the header line
        if '\t' not in stripped_line:  # Check if the line is a continuation
            temp_line += " \t" + stripped_line  # Continue the previous record
        else:
            if temp_line:
                processed_lines.append(temp_line)  # Add the completed record
            temp_line = stripped_line  # Start a new record

    if temp_line:  # Append the last line if it was in progress
        processed_lines.append(temp_line)

    # Split the processed lines into columns and remove whitespaces
    data_aux = [line.split('\t') for line in processed_lines]
    data = [[value.strip() for value in line] for line in data_aux]

    # Convert to DataFrame
    columns = ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3']
    joints_matrix_index = pd.DataFrame(data, columns=columns)

    # Transform the values to integers and adjust indexing
    def transform_value(value):
        if value == '0':  # restrained nodes are labeled as 0
            return np.nan
        else:
            return int(int(value) - 1)  # python index start at 0

    for col in columns[1:]:  # Skip the first column 'Joint_Label'
        joints_matrix_index[col] = joints_matrix_index[col].apply(
            transform_value)

    return joints_matrix_index


def get_mass_stiffness_matrix(FilePath):
    """
    Function Duties:
        Read the mass or stiffness matrix from the .TXM or .TXK file
    Input:
        FilePath: Path to the .TXM (for mass) or .TXK (for stiffness) file
    Output:
        matrix: Mass or stiffness matrix in NumPy array format
    """
    # Read the stiffness matrix
    with open(FilePath, 'r') as file:
        lines = file.readlines()

    data_aux = [line.split('\t') for line in lines[1:]]
    data = [[value.strip() for value in line] for line in data_aux]

    # convert to numpy array
    data = [[int(row[0]), int(row[1]), float(row[2])] for row in data]

    # Determine the shape of the matrix (max row and column indices)
    max_row = max(row[0] for row in data)
    max_col = max(row[1] for row in data)

    # Create an empty NumPy array with the determined shape
    matrix = np.zeros((max_row, max_col))

    # Populate the matrix with the values from the data list
    for row, col, value in data:
        matrix[row-1, col-1] = value  # Adjust for zero-based indexing
        matrix[col-1, row-1] = value  # Symmetric matrix

    return matrix


def add_joints_to_channel_coordinates(channel_coordinates, joint_coordinates, round_coordinates=True):
    """
    Function Duties:
        Add joint id to the dictionary of channel coordinates
    Input:
        channel_coordinates: dictionary with channel coordinates (from measured data)
        joint_coordinates: dictionary with joint coordinates (from SAP2000 model)
        round_coordinates: boolean to round coordinates to 6 significant digits
    Output:
        channel_coordinates: dictionary with channel coordinates and joint id
    """
    # Retrieve the joint for the channel coordinates
    for ch in list(channel_coordinates):
        coord_channel = [channel_coordinates[ch][i] for i in ['x', 'y', 'z']]
        if round_coordinates:
            coord_channel = [round_6_sign_digits(
                coord) for coord in coord_channel]
        for joint in list(joint_coordinates):
            coord_joint = [joint_coordinates[joint][i]
                           for i in ['x', 'y', 'z']]
            if round_coordinates:
                coord_joint = [round(coord, 6) for coord in coord_joint]
            if coord_channel == coord_joint:
                channel_coordinates[ch]['joint'] = joint
                continue

    # Check if all channels have a joint associated
    for ch in list(channel_coordinates):
        if 'joint' not in channel_coordinates[ch]:
            warnings.warn(f"Channel {ch} has no joint associated")

    return channel_coordinates


def add_noise_to_frequencies(frequencies, sigma_f):
    """
    Function Duties:
        Add noise to frequencies from SAP2000 model
    Input:
        frequencies: dict
            Frequencies from SAP2000 model
        sigma_f: float
            Standard deviation of the noise for the frequencies
    Output:
        measured_frequencies: dict
            Frequencies with noise
    Remark:
        Standard deviation of the noise is proportional to the frequency value
    """
    measured_frequencies = dict()
    for mode_label in list(frequencies):
        random_noise = frequencies[mode_label]['Frequency'] * \
            np.random.normal(0, sigma_f)
        measured_frequencies[mode_label] = frequencies[mode_label]['Frequency'] + random_noise
    return measured_frequencies


def add_noise_to_mode_shapes(disp_modeshapes, channel_coordinates, sigma_f, w, scale_modes=True):
    """
    Function Duties:
        Add noise to mode shapes from SAP2000 model
    Input:
        disp_modeshapes: dict
            Mode shapes from SAP2000 model
        channel_coordinates: dict
            Channel coordinates from measured data
        sigma_f: float
            Standard deviation of the noise for the frequencies
        w: float
            Scaling factor for standard deviation of modes
    Output:
        measured_disp_modeshapes: dict
            Mode shapes with noise
    Remark:
        Standard deviation of the noise is proportional to the mode norm and sigma_f/np.sqrt(w)
        The mode shapes are scaled by a random factor
    """
    measured_disp_modeshapes = dict()
    for mode_label in list(disp_modeshapes):
        model_disp_modeshape = list()
        ch_id = list()
        for ch in channel_coordinates:
            if 'joint' in channel_coordinates[ch]:
                joint = channel_coordinates[ch]['joint']
                dof = channel_coordinates[ch]['ax']
                joint_id = disp_modeshapes[mode_label]['Joint_id']
                measured_value = disp_modeshapes[mode_label][dof][joint_id.index(
                    joint)]
                model_disp_modeshape.append(measured_value)
                ch_id.append(ch)
        random_noise = np.linalg.norm(np.array(model_disp_modeshape)) * np.random.normal(
            0, sigma_f/np.sqrt(w), len(model_disp_modeshape))
        measured_mode = np.array(model_disp_modeshape) + random_noise
        if scale_modes:  # Add random scaling
            measured_mode *= np.random.uniform(-5, 5)
        measured_disp_modeshapes[mode_label] = {
            'ch_id': copy.deepcopy(ch_id),
            'coord': copy.deepcopy(list(measured_mode))
            }

    return measured_disp_modeshapes


def adapt_modal_results(modal_results, channel_coordinates):
    """
    Function Duties:
        Adapt the measured displacement mode shapes to the same format as the model mode shapes,
        filling with nan values where there are no sensor measurements.
    Input:
        - modal_results: dictionary with the measured displacement mode shapes
        - channel_coordinates: dictionary with the coordinates of the channels
    Output:
        - measured_disp_modeshapes: dictionary with the adapted measured displacement mode shapes
    """
    measured_disp_modeshapes = dict()
    for mode_label in list(modal_results):
        measured_disp_modeshapes[mode_label] = dict()
        for label in ['U1', 'U2', 'U3', 'R1', 'R2', 'R3', 'Joint_id']:
            measured_disp_modeshapes[mode_label][label] = list()
        for ch in modal_results[mode_label]['disp_modeshapes']['ch_id']:
            joint = channel_coordinates[ch]['joint']
            if joint not in measured_disp_modeshapes[mode_label]['Joint_id']:
                measured_disp_modeshapes[mode_label]['Joint_id'].append(joint)
                for label in ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']:
                    measured_disp_modeshapes[mode_label][label].append(np.nan)
            index_joint = measured_disp_modeshapes[mode_label]['Joint_id'].index(
                joint)
            dof = channel_coordinates[ch]['ax']
            index_modal = modal_results[mode_label]['disp_modeshapes']['ch_id'].index(
                ch)
            displacement = modal_results[mode_label]['disp_modeshapes']['coord'][index_modal]
            measured_disp_modeshapes[mode_label][dof][index_joint] = displacement

    return measured_disp_modeshapes


def get_observation_matrix(disp_modeshapes_measured, joints_matrix_index, n_DOFS_observed, n_DOFS_model):
    """
    Function Duties:
        Get the observation matrix L for the system
    Inputs:
        disp_modeshapes_measured: dictionary with the measured displacement mode shapes
        joints_matrix_index: DataFrame with the joint matrix index
        n_DOFS_observed: number of observed DOFs
        n_DOFS_model: number of model DOFs
    Output:
        L: observation matrix
    Remark:
        i) n_DOFS_observed, n_DOFS_model must be in agreement with the variables
            in disp_modeshapes_measured and joints_matrix_index
        ii) The function assumes that all modes have the same number of measured
            points (as well as the same number of model points)
    """
    L = np.zeros((n_DOFS_observed, n_DOFS_model))
    mode_label = list(disp_modeshapes_measured.keys())[0]
    all_eq = list()
    for joint in disp_modeshapes_measured[mode_label]['Joint_id']:
        eq = joints_matrix_index[joints_matrix_index['Joint_Label'] == joint]
        for dof in ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']:
            eq_number = eq[dof].values[0]
            index = disp_modeshapes_measured[mode_label]['Joint_id'].index(joint)
            if (np.isnan(eq_number)) | (np.isnan(disp_modeshapes_measured[mode_label][dof][index])):
                continue
            all_eq.append((joint, dof, int(eq_number)))
    all_eq_sorted = sorted(all_eq, key=lambda x: x[2])
    for i in range(len(all_eq_sorted)):
        L[i, all_eq_sorted[i][2]] = 1

    return L


def get_modeshape_matrix_from_dict(disp_modeshapes, stiffness_matrix, joints_matrix_index):
    """
    Function Duties:
        Get the matrix of mode shapes from the dictionary of mode shapes
    Input:
        disp_modeshapes: dictionary with the mode shapes
        stiffness_matrix: stiffness matrix of the system
        joints_matrix_index: DataFrame with the joint matrix index
    Output:
        phi_matrix: matrix of mode shapes (n_DOFS x n_modes)
    Remark:
        If disp_modeshapes contains less measured points than the model,
            nan values are introduced in the matrix
    """
    phi_matrix = np.full(
        (np.shape(stiffness_matrix)[0], len(list(disp_modeshapes))), np.nan)
    for n_mode, n_mode_label in enumerate(list(disp_modeshapes)):
        vector = np.full(np.shape(phi_matrix)[0], np.nan)
        mode_values = disp_modeshapes[n_mode_label]
        joint_id = disp_modeshapes[n_mode_label]['Joint_id']
        dofs = list(mode_values)[0:6]
        for joint in joint_id:
            eq = joints_matrix_index[joints_matrix_index['Joint_Label'] == joint]
            for dof in dofs:
                eq_number = eq[dof].values[0]
                if not np.isnan(eq_number):
                    vector[int(eq_number)] = mode_values[dof][joint_id.index(joint)]
        phi_matrix[:, n_mode] = vector

    return phi_matrix


def round_complex_array(arr, decimals=3):
    """
    Rounds a complex array to a given number of decimals
    """
    return np.round(arr.real, decimals) + 1j * np.round(arr.imag, decimals)


def format_complex(c, decimals=2):
    """
    Converts a complex number into a string with 2 decimal places
    """
    if decimals not in [1, 2, 3, 4]:
        decimals = 2
        warnings.warn('Decimals must be 1, 2, 3 or 4. Setting decimals=2', UserWarning)

    if decimals == 1:
        real_part = f"{c.real:.1f}"
        imag_part = f"{c.imag:.1f}j"
    elif decimals == 2:
        real_part = f"{c.real:.2f}"
        imag_part = f"{c.imag:.2f}j"
    elif decimals == 3:
        real_part = f"{c.real:.3f}"
        imag_part = f"{c.imag:.3f}j"
    elif decimals == 4:
        real_part = f"{c.real:.4f}"
        imag_part = f"{c.imag:.4f}j"
    if c.imag >= 0:
        imag_part = f"+{imag_part}"

    return f"{real_part}{imag_part}"


def mode_matching_mac(frequencies, Psi, Psi_oma_real):
    """
    Function Duties:
        Simplest mode matching based on the MAC value.
    Input:
        frequencies: dict of type {"Mode_1": freq_1, "Mode_2": freq_2, ...} for FEModel
        Psi: np.array
            model modes
        Psi_oma_real: np.array
            oma modes (real extracted modes)
    """
    frequencies_matched = np.zeros((np.shape(Psi_oma_real)[1]))
    Psi_matched = np.zeros((np.shape(Psi_oma_real)[0], np.shape(Psi_oma_real)[1]))
    mac = np.zeros((np.shape(Psi_oma_real)[1], np.shape(Psi)[1]))
    mode_matching = dict()
    for i in range(np.shape(Psi_oma_real)[1]):
        mac[i, :] = MaC(Psi_oma_real[:, i], Psi)
        Psi_matched[:, i] = Psi[:, np.argmax(mac[i, :])]
        frequencies_matched[i] = [frequencies[i] for i in frequencies][np.argmax(mac[i, :])]
        mode_matching['Mode_' + str(i+1)] = {'OMA': str(i+1),
                                             'FEM': list(frequencies)[np.argmax(mac[i, :])]}

    return frequencies_matched, Psi_matched, mode_matching


def safe_eval(value):
    """
    Function Duties:
        Safely evaluate a string as a literal or expression
    """
    try:
        # Try to evaluate the value as a literal
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If that fails, use eval to handle expressions like 10**6
        return eval(value)


def read_parameters(file_path):
    """
    Function Duties:
        Read bayesian_inference_parameters.txt file
    """
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Remove comments and strip whitespace
            line = line.split('#')[0].strip()
            if line:  # Skip empty lines
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if value == 'None':
                    parameters[key] = None
                elif 'filename' in key:
                    parameters[key] = value
                else:
                    parameters[key] = safe_eval(value)
    return parameters


def read_matrices_h5py(h5py_filename, groups):
    """
    Read matrices from a h5py file
    """
    with h5py.File(h5py_filename, 'r') as h5py_file:
        matrices = dict()
        for group in groups:
            matrices[group] = h5py_file[group][:]
    return matrices


def kill_process(process_name):
    """
    Function Duties:
        Kill a process if it is running
    Input:
        process_name: str
    Return:
        None
    """
    result = subprocess.run(["tasklist"], capture_output=True, text=True)
    if process_name.lower() in result.stdout.lower():
        print(f"The process '{process_name}' is running. Attempting to terminate it.")
        subprocess.run(["taskkill", "/f", "/im", process_name])


def kill_process_advanced(process_name):
    """Kills the specified process and its subprocesses."""
    found_process = False
    for proc in psutil.process_iter():
        try:
            if process_name.lower() in proc.name().lower():
                found_process = True
                print(f"Killing process '{proc.name()}' with PID {proc.pid}")
                proc.kill()
                proc.wait()  # Wait for the process to fully terminate

                # Check for any child processes and kill them
                for child in proc.children(recursive=True):
                    print(f"Killing subprocess '{child.name()}' with PID {child.pid}")
                    child.kill()
                    child.wait()

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Error: {e}")

    if not found_process:
        print(f"The process '{process_name}' is not running.")


def monitor_process(process, timeout_duration):
    """
    Function Duties:
        Process monitoring
    Output:
        True: if the process finishes before the timeout
        False: otherwise
    """
    start_time = time.time()
    while process.is_alive():
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_duration:
            print("Timeout reached. Terminating process...")
            process.terminate()  # Terminate the process
            process.join()  # Wait for process termination
            print("Process terminated due to timeout.")
            return False
        time.sleep(0.05)  # Check every 0.05 seconds
    return True


def sap2000_get_modal_response_task(input_data, SapModel, result_queue):
    """Process function to execute sap2000.get_modal_response."""
    modal_results = sap2000.get_modal_response(input_data, SapModel)
    result_queue.put(modal_results)  # Send results back to the main process


def save_state(state, filepath):
    """Save the current state to a file."""
    with open(filepath, 'w') as f:
        json.dump(state, f)
    print(f"State saved to {filepath}")


def load_state(filepath):
    """Load the state from a file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        print(f"State loaded from {filepath}")
        return state
    return None


def remove_file(file_path):
    """Remove a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed file: {file_path}")
