
"""
File Description:
    Contains helpers that use SAP2000 OAPI

Created on 2024/02/06 by asanchezlc
"""
from collections import defaultdict
import comtypes.client
import numpy as np
import warnings

import helpers.outils as outils


def raise_warning(process_name, ret) -> None:
    """
    Raise warning if ret=1
    """
    if ret == 1:
        message = process_name + ' was not properly retrieved in SAP2000'
        warnings.warn(message, UserWarning)


def app_start(use_GUI=True):
    """
    Function duties:
        Starts sap2000 application
    """
    # create API helper object
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
    mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")

    # start SAP2000 application
    mySapObject.ApplicationStart(3, use_GUI, "")

    return mySapObject


def open_file(mySapObject, FilePath):
    """
    Function duties:
        Once the application has started, open an existing SAP2000 file
    """
    # create SapModel object
    SapModel = mySapObject.SapModel

    # initialize model
    ret = SapModel.InitializeNewModel()
    raise_warning('Initialize SAP2000', ret)

    # open existing file
    ret = SapModel.File.OpenFile(FilePath)
    raise_warning('Open file', ret)

    return SapModel


def unlock_model(SapModel, lock=False) -> None:
    """
    Function duties:
        If lock=False: unlocks model
        Else: locks model
    """
    ret = SapModel.SetModelIsLocked(False)
    if lock:
        raise_warning("Unlock", ret)
    else:
        raise_warning("Lock", ret)


def set_ISunits(SapModel) -> None:
    """
    Function Duties:
        Sets units in International Sistem
    """
    N_m_C = 10
    ret = SapModel.SetPresentUnits(N_m_C)
    raise_warning('setting units', ret)


def set_materials(material_dict, SapModel) -> None:
    """
    Function Duties:
        Changes materials properties
    Input:
        material_dict: dictionary as follow:
            material_dict{mat1_dict, mat2_dict...}
            mat1_dict = {'E': E1, 'u': u1, 'a': a1}
            Properties must be defined in International System
    """
    for material in list(material_dict):
        # 1. we read material properties:
        # A) E, u, a, g
        E_0, u_0, a_0, g_0 = 0, 0, 0, 0
        output = SapModel.PropMaterial.GetMPIsotropic(
            material, E_0, u_0, a_0, g_0)
        [E_0, u_0, a_0, g_0, ret] = output
        raise_warning(f'{material} properties reading', ret)

        E, u, a, _ = float(E_0), float(u_0), float(a_0), float(g_0)

        # B) rho
        w_0, rho_0 = 0, 0
        output = SapModel.PropMaterial.GetWeightAndMass(material, w_0, rho_0)
        [type_data, rho_0, ret] = output
        raise_warning(f'{material} density reading', ret)

        rho = float(rho_0)

        # 2. We update available properties
        # A) Retrieve properties from dictionary
        mat_list = [E, u, a, rho]
        for i, mat_prop in enumerate(['E', 'u', 'a', 'rho']):
            if mat_prop in list(material_dict[material]):
                mat_list[i] = material_dict[material][mat_prop]

        [E, u, a, rho] = mat_list

        # B) Set rho
        type_data = 2  # 1: weight/volume; 2: mass/volume
        ret = SapModel.PropMaterial.SetWeightAndMass(material, type_data, rho)
        raise_warning(f'{material} rho assignment', ret)

        # C) Set E, u, g, a
        ret = SapModel.PropMaterial.SetMPIsotropic(material, E, u, a)
        raise_warning(f'{material} E, u, a assignment', ret)


def set_jointsprings(joint_spring_dict, SapModel) -> None:
    """
    Function Duties:
        Sets springs to joints included in support_name group.
    Input:
        joint_spring_dict: dictionary whose keys are names of joint groups.
        For each key, k array (of 6 elements) is defined:
            k[0], k[1], k[2] = U1 [F/L], U2 [F/L], U3 [F/L]
            k[3], k[4], k[5] = R1 [FL/rad], R2 [FL/rad], R3 [FL/rad]
    """
    for name in list(joint_spring_dict):
        k = joint_spring_dict[name]
        ItemType = 1  # 1 for group
        IsLocalCSys = True  # spring assignments are in the point local coordinate system
        Replace = True  # delete prior assignment
        output = SapModel.PointObj.SetSpring(
            name, k, ItemType, IsLocalCSys, Replace)
        [_, ret] = output
        raise_warning('Spring assignment', ret)


def set_framereleases(frame_releases_dict, SapModel) -> None:
    """
    Function Duties:
        Applies releases to frame groups included in frame_releases_dict
    Input:
        frame_releases_dict:
            Dictionary whose keys are names of frame groups. For each key:
                ii: list of 6 boolean elements for i extreme. Each element corresponds
                    to U1, U2, U3, R1, R2, R3: True if released, False elsewhere)
                jj: list of 6 boolean elements for j extreme. Each element corresponds
                    to U1, U2, U3, R1, R2, R3: True if released, False elsewhere)
                StartValue: list of 6 elements, corresponding to partial fixity in ext. i;
                    U1 [F/L], U2 [F/L], U3 [F/L], R1 [FL/rad], R2 [FL/rad], R3 [FL/rad]
                EndValue: analogous to StartValue; extreme j
    """
    for name in list(frame_releases_dict):
        ii = frame_releases_dict[name]['ii']
        jj = frame_releases_dict[name]['jj']
        StartValue = frame_releases_dict[name]['StartValue']
        EndValue = frame_releases_dict[name]['EndValue']
        ItemType = 1  # 1 means group name
        output = SapModel.FrameObj.SetReleases(
            name, ii, jj, StartValue, EndValue, ItemType)
        [ii, jj, StartValue, EndValue, ret] = output
        raise_warning('Partial fixity assignment', ret)


def run_analysis(SapModel) -> None:
    """
    Function Duties:
        Runs analysis
    """
    ret = SapModel.Analyze.RunAnalysis()
    raise_warning('Run analysis', ret)


def get_modalfrequencies(SapModel):
    """
    Function duties:
        Retrieve frequency associated to each mode
    Input:
        NumberResults: Number of modes want to be retrieved
    Output:
        frequency_results: dictionary with results
    """
    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    NumberResults = 0
    LoadCase, StepType, StepNum, Period = [], [], [], []
    Frequency, CircFreq, EigenValue = [], [], []

    output = SapModel.Results.ModalPeriod(NumberResults, LoadCase, StepType,
                                          StepNum, Period, Frequency, CircFreq,
                                          EigenValue)
    [NumberResults, _, _, StepNum, _, Frequency, _, _, ret] = output
    raise_warning('Get displacement mode shapes', ret)

    StepNum = [int(i) for i in StepNum]  # modeshape id in integer format

    frequency_results = dict()
    for i, mode_id in enumerate(StepNum):
        mode_label = 'Mode_' + str(mode_id)
        frequency_results[mode_label] = dict()
        frequency_results[mode_label]['Frequency'] = Frequency[i]

    return frequency_results


def getnames_point_elements(Name_points_group, Name_elements_group, SapModel):
    """
    Function duties:
        Retrieves all point labels
    Input:
        Name_points_group: This name comes from SAP2000! (Check that the group exists)
        Name_elements_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        all_points: list with all points
        all_elements: list with all points
    Remark: bug found:
        Once the .sdb has been file we change it manually (from the SAP2000
        interface) this function does not work well, and the table has to be
        manually opened before executing this function
    """
    # Run the model if it is not locked
    if not (SapModel.GetModelIsLocked()):
        run_analysis(SapModel)

    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # A) Get all names of points
    # group element (if instead of group we want to get elements, then 1)
    GroupElm = 2
    Obj = 1  # I don't know why is it necessary (but it works with any number)
    LoadCase, StepType, StepNum = [], [], []
    U1, U2, U3 = [], [], []
    R1, R2, R3 = [], [], []

    output = SapModel.Results.ModeShape(Name_points_group,
                                        GroupElm, Obj, LoadCase,
                                        StepType, StepNum, U1,
                                        U2, U3, R1, R2, R3)
    [_, Point, _, _, _, _, _, _, _, _, _, _, ret] = output
    raise_warning('Get all points names', ret)

    # B) Get all names of elements and element stations
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [_, Element, _, ElementStat, _, _, _, _, _, _, _, _, _, _, ret] = output
    raise_warning('Get all element names', ret)

    all_points = list(set(Point))
    all_elements = list(set(Element))
    all_elements_stat = list(set(ElementStat))

    return all_points, all_elements, all_elements_stat


def get_displmodeshapes(Name_points_group, SapModel):
    """
    Function duties:
        Retrieves displacement mode shapes for a given group of points
    Input:
        name_points_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        joint_modeshape_results: dictionary of results
    Remark: bug found:
        If we manually visualize results (i.e., show tables) in SAP2000 (from
        the interface) this function stop working well, and the table has to be
        manually opened before executing this function
    """
    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # Get results from SAP2000
    # group element (if instead of group we want to get elements, then 1)
    GroupElm = 2
    Obj = 1  # I don't know why is it necessary (but it works with any number)
    LoadCase = "MODAL"
    StepType, StepNum = [], []
    U1, U2, U3 = [], [], []
    R1, R2, R3 = [], [], []

    output = SapModel.Results.ModeShape(Name_points_group,
                                        GroupElm, Obj, LoadCase,
                                        StepType, StepNum, U1,
                                        U2, U3, R1, R2, R3)

    raise_warning('Get displacement mode shapes', ret)

    [NumberResults, Obj, Elm, LoadCase, StepType,
        StepNum, U1, U2, U3, R1, R2, R3, ret] = output
    # [NumberResults, GroupElm, _, _, _, StepNum, U1, U2, U3, R1, R2, R3, ret] = output

    # Save results into dictionary
    StepNum = [int(i) for i in StepNum]  # modeshape id in integer format
    joint_modeshape_results = dict()

    for mode in list(set(StepNum)):
        mode_label = "Mode_" + str(mode)
        joint_modeshape_results[mode_label] = dict()
        mode_id = [mode == step for step in StepNum]

        joint_modeshape_results[mode_label]['U1'] = [
            _u for _i, _u in enumerate(list(U1)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['U2'] = [
            _u for _i, _u in enumerate(list(U2)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['U3'] = [
            _u for _i, _u in enumerate(list(U3)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['R1'] = [
            _r for _i, _r in enumerate(list(R1)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['R2'] = [
            _r for _i, _r in enumerate(list(R2)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['R3'] = [
            _r for _i, _r in enumerate(list(R3)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['Joint_id'] = [
            _obj for _i, _obj in enumerate(list(Obj)) if mode_id[_i]]

    return joint_modeshape_results


def get_pointcoordinates(all_points, SapModel, round_coordinates=True):
    """
    Function duties:
        Returns a dictionary with x-y-z coordinates for each point
        in all_points list
    Input:
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
    """
    pointcoord_dict = dict()
    for point in outils.sort_list_string(all_points):
        pointcoord_dict[point] = dict()
        x, y, z = 0, 0, 0
        output = SapModel.PointElm.GetCoordCartesian(point, x, y, z)
        [x, y, z, ret] = output
        raise_warning('Point coordinates', ret)

        if round_coordinates:
            x = outils.round_6_sign_digits(x)
            y = outils.round_6_sign_digits(y)
            z = outils.round_6_sign_digits(z)

        pointcoord_dict[point]['x'] = x
        pointcoord_dict[point]['y'] = y
        pointcoord_dict[point]['z'] = z

    return pointcoord_dict


def get_frameconnectivity(all_points, all_elements,
                          SapModel, all_points_coord={}):
    """
    Function duties:
        For each frame, it gives initial and end point names;
        If all_points_coord dictionary is introduced, also coordinates of each point are provided
    Input:
        all_points: list with all points
        all_elements: list with all elements
        all_points_coord: if provided, dict containing x, y, z coordinates for each point
            (this dictionary comes from sap2000_getpointcoordinates function)
    """
    if len(all_points_coord) > 0:
        coord_defined = True
    else:
        coord_defined = False

    frames_dict = dict()
    key_ini, key_end = 'Point_0', 'Point_f'
    keys = [key_ini, key_end]
    for element in outils.sort_list_string(all_elements):
        frames_dict[element] = {key: None for key in keys}

    for PointName in all_points:
        NumberItems = 0
        ObjectType, ObjectName, PointNumber = [], [], []
        output = SapModel.PointObj.GetConnectivity(PointName, NumberItems,
                                                   ObjectType, ObjectName, PointNumber)
        [NumberItems, ObjectType, ObjectName, PointNumber, ret] = output
        raise_warning('Get connectivity', ret)

        # element_joint_connect = dict()
        ObjectType, ObjectName, PointNumber = list(
            ObjectType), list(ObjectName), list(PointNumber)
        FRAME_ID = 2  # stablished by SAP2000
        OBJECT_INI, OBJECT_END = 1, 2  # stablished by SAP2000

        # Retrieve frames (other elements could be connected)
        frames = [name for i, name in enumerate(
            ObjectName) if ObjectType[i] == FRAME_ID]

        for i, frame in enumerate(frames):
            coord = {}
            if PointNumber[i] == OBJECT_INI:
                if frames_dict[frame][key_ini] is None:
                    if coord_defined:
                        coord = all_points_coord[PointName]
                    frames_dict[frame][key_ini] = {
                        **{'PointName': PointName}, **coord}
                else:
                    if frames_dict[frame][key_ini] != PointName:
                        message = f'{frame} frame has 2 different {key_ini} assigments: ({PointName} and {frames_dict[frame][key_ini]}) '
                        warnings.warn(message, UserWarning)
            elif PointNumber[i] == OBJECT_END:
                if frames_dict[frame][key_end] is None:
                    if coord_defined:
                        coord = all_points_coord[PointName]
                    frames_dict[frame][key_end] = {
                        **{'PointName': PointName}, **coord}
                else:
                    if frames_dict[frame][key_end] != PointName:
                        message = f'{frame} frame has 2 different {key_end} assigments: ({PointName} and {frames_dict[frame][key_end]}) '
                        warnings.warn(message, UserWarning)

    return frames_dict


def get_modalforces(Name_elements_group, SapModel,
                    average_values=True, round_coordinates=True,
                    considered_modes=None):
    """
    Input:
        Name_elements_group: name of the group containing elements whose
            forces will be retrieved
                Remark: This name is defined in SAP2000 .sdb model!!
        SapModel: SAP2000 model
        average_values: if True, values in elements with the same x value are averaged
            (Set to False when comparing with SAP2000 interface tables)
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
        considered_modes:
            if None, all modes are considered
            if list, only modes in the list are considered
    Output:
        results: dictionary containing information for each mode as follow:
            Mode_i: element_1 to element_N:
                element_1: P, M2 and M3 per each x coordinate within the element
    """
    # Select modal case
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # Read information from SAP2000
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [NumResults, Element, Station, Mesh, MeshStation, LoadCase,
        StepType, StepNum, P, V2, V3, T, M2, M3, ret] = output
    raise_warning('Get element forces', ret)

    StepNum = list(StepNum)
    Element = list(Element)
    if round_coordinates:
        Station = list([outils.round_6_sign_digits(i) for i in Station])
        MeshStation = list([outils.round_6_sign_digits(i) for i in MeshStation])
    P, M2, M3 = list(P), list(M2), list(M3)

    # Save information into a dictionary
    StepNum_unique = list(set(StepNum))
    StepNum_unique.sort()
    Element_unique = outils.sort_list_string(list(set(Element)))
    if considered_modes is not None:
        StepNum_unique = [mode for mode in StepNum_unique if mode in considered_modes]

    results = dict()
    for mode in StepNum_unique:
        mode_label = 'Mode_' + str(int(mode))
        results[mode_label] = dict()
        for element in Element_unique:
            element_label = 'Element_' + str(int(element))
            results[mode_label][element_label] = dict()

            # Bool list for for specific mode and element
            id_mode_element = [el == element and StepNum[i]
                               == mode for i, el in enumerate(Element)]

            # Coordinates and forces retrievement
            station_mode_element = [st for i, st in enumerate(
                Station) if id_mode_element[i]]
            p = [force for i, force in enumerate(P) if id_mode_element[i]]
            m2 = [force for i, force in enumerate(M2) if id_mode_element[i]]
            m3 = [force for i, force in enumerate(M3) if id_mode_element[i]]

            if average_values:
                # values in elements with the same x value are averaged
                x = outils.sort_list_string(list(set(station_mode_element)))
                p_def, m2_def, m3_def = list(), list(), list()
                for i in x:
                    average_id = [i == x for x in station_mode_element]
                    p_aux = [p[j] for j, _ in enumerate(p) if average_id[j]]
                    m2_aux = [m2[j] for j, _ in enumerate(m2) if average_id[j]]
                    m3_aux = [m3[j] for j, _ in enumerate(m3) if average_id[j]]
                    p_def.append(np.mean(p_aux))
                    m2_def.append(np.mean(m2_aux))
                    m3_def.append(np.mean(m3_aux))
            else:
                # retrieve left-hand value in elements with the same x value (for validating purposes)
                x_coordinate_id = [st != station_mode_element[i-1] for i, st in enumerate(station_mode_element)]
                x = [x for i, x in enumerate(station_mode_element) if x_coordinate_id[i]]
                p_def = [force for i, force in enumerate(p) if x_coordinate_id[i]]
                m2_def = [force for i, force in enumerate(m2) if x_coordinate_id[i]]
                m3_def = [force for i, force in enumerate(m3) if x_coordinate_id[i]]

            mesh = [f'{element}.{mesh_id}' for mesh_id in range(len(x))]

            # Save into results dictionary
            results[mode_label][element_label]['x'] = x
            results[mode_label][element_label]['P'] = p_def
            results[mode_label][element_label]['M2'] = m2_def
            results[mode_label][element_label]['M3'] = m3_def
            results[mode_label][element_label]['Mesh_id'] = mesh

    return results


def get_modalforces_timehistory(Name_elements_group, SapModel,
                                loadcase_name, average_values=True,
                                round_coordinates=True,
                                selected_elements=None):
    """
    Input:
        Name_elements_group: name of the group containing elements whose
            forces will be retrieved
                Remark: This name is defined in SAP2000 .sdb model!!
        SapModel: SAP2000 model
        loadcase_name: name of the time history load case
        average_values: if True, values in elements with the same x value are averaged
            (Set to False when comparing with SAP2000 interface tables)
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
        considered_modes:
            if None, all modes are considered
            if list, only modes in the list are considered
        selected_elements: a list of elements to be considered (if None, all elements are considered)
            it is used to filter the results (much smaller dictionary)
    Output:
        results: dictionary containing information for the time history as follow:
            Mode_i: element_1
            element_1 to element_N:
                x: x coordinates
                P, M2 and M3: arrays whose i,j elements refer to:
                    i: x coordinate
                    j: time step (from StepNum)
        StepNum: list with time steps (same for all elements)
    """
    # Select modal case
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput(loadcase_name)
    raise_warning('Select modal case', ret)

    # Change display table options (only way I found to show step-by-step results)
    Envelopes, Step_by_Step = 1, 2
    BaseReactionGX, BaseReactionGY, BaseReactionGZ = 0, 0, 0
    IsAllModes, StartMode, EndMode = False, 1, 1
    IsAllBucklingModes, StartBuckingMode, EndBucklingMode = False, 1, 1
    ModalHistory, DirectHistory, NonLinearStatic = Step_by_Step, Step_by_Step, Step_by_Step
    MultistepStaticStatic, SteadyState = Step_by_Step, Envelopes
    SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing = 1, 1, 1, 1
    ret = SapModel.DatabaseTables.SetTableOutputOptionsForDisplay(BaseReactionGX, BaseReactionGY, BaseReactionGZ, IsAllModes, StartMode, EndMode, IsAllBucklingModes, StartBuckingMode, EndBucklingMode, ModalHistory, DirectHistory, NonLinearStatic, MultistepStaticStatic, SteadyState, SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing)
    raise_warning('Display tables - options', ret)

    # Read information from SAP2000
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [NumResults, Element, Station, Mesh, MeshStation, LoadCase,
        StepType, StepNum, P, V2, V3, T, M2, M3, ret] = output
    raise_warning('Get element forces', ret)

    # Save into a dictionary (enhanced version of get_modalforces function for faster performance)
    Element, Station, StepNum, LoadCase = np.array(Element), np.array(Station), np.array(StepNum), np.array(LoadCase)
    P, M2, M3 = np.array(P), np.array(M2), np.array(M3)

    # Round if necessary
    if round_coordinates:
        Station = np.vectorize(outils.round_6_sign_digits)(Station)
        MeshStation = np.vectorize(outils.round_6_sign_digits)(MeshStation)
        StepNum = np.vectorize(outils.round_6_sign_digits)(StepNum)

    # Filtering for the specific load case
    loadcase_mask = (LoadCase == loadcase_name)
    Element = Element[loadcase_mask]
    Station = Station[loadcase_mask]
    StepNum = StepNum[loadcase_mask]
    P = P[loadcase_mask]
    M2 = M2[loadcase_mask]
    M3 = M3[loadcase_mask]

    # Unique Elements and Step Numbers
    Element_unique = np.array(outils.sort_list_string(list(set(Element))))
    StepNum_unique = np.unique(StepNum)

    results = defaultdict(lambda: {'x': [], 'P': [], 'M2': [], 'M3': []})

    # Group data by element
    if selected_elements is None:
        selected_elements = Element_unique

    for element in selected_elements:
        element_mask = (Element == element)
        station_element = Station[element_mask]
        step_element = StepNum[element_mask]
        p_element = P[element_mask]
        m2_element = M2[element_mask]
        m3_element = M3[element_mask]

        if average_values:
            x_unique, x_indices = np.unique(station_element, return_inverse=True)
            num_stations = len(x_unique)
            num_steps = len(StepNum_unique)

            p_def = np.full((num_stations, num_steps), np.nan)
            m2_def = np.full((num_stations, num_steps), np.nan)
            m3_def = np.full((num_stations, num_steps), np.nan)

            for i_step, step in enumerate(StepNum_unique):
                step_mask = (step_element == step)
                p_step = p_element[step_mask]
                m2_step = m2_element[step_mask]
                m3_step = m3_element[step_mask]
                station_step = x_indices[step_mask]

                for i_station in range(num_stations):
                    station_mask = (station_step == i_station)
                    if station_mask.any():
                        p_def[i_station, i_step] = np.mean(p_step[station_mask])
                        m2_def[i_station, i_step] = np.mean(m2_step[station_mask])
                        m3_def[i_station, i_step] = np.mean(m3_step[station_mask])

            element_label = f'Element_{int(element)}'
            results[element_label]['x'] = x_unique.tolist()
            results[element_label]['P'] = p_def.tolist()
            results[element_label]['M2'] = m2_def.tolist()
            results[element_label]['M3'] = m3_def.tolist()
            # results[element_label]['Mesh_id'] = [f'{element}.{mesh_id}' for mesh_id in range(len(x_unique))]

    # Convert defaultdict to dict
    results = dict(results)

    # Come back to default display table options
    Envelopes, Step_by_Step = 1, 2
    BaseReactionGX, BaseReactionGY, BaseReactionGZ = 0, 0, 0
    IsAllModes, StartMode, EndMode = False, 1, 1
    IsAllBucklingModes, StartBuckingMode, EndBucklingMode = False, 1, 1
    ModalHistory, DirectHistory, NonLinearStatic = Envelopes, Envelopes, Envelopes
    MultistepStaticStatic, SteadyState = Envelopes, Envelopes
    SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing = 1, 1, 1, 1
    ret = SapModel.DatabaseTables.SetTableOutputOptionsForDisplay(BaseReactionGX, BaseReactionGY, BaseReactionGZ, IsAllModes, StartMode, EndMode, IsAllBucklingModes, StartBuckingMode, EndBucklingMode, ModalHistory, DirectHistory, NonLinearStatic, MultistepStaticStatic, SteadyState, SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing)
    raise_warning('Display tables - options', ret)

    return results, StepNum_unique.tolist()


def get_loadcaseforces(Name_elements_group, SapModel,
                       average_values=True, round_coordinates=True,
                       load_case='DEAD'):
    """
    Function duties:
        This function is very similar to get_modalforces (in fact, both could be
            merged)
    Input:
        Name_elements_group: name of the group containing elements whose
            forces will be retrieved
                Remark: This name is defined in SAP2000 .sdb model!!
        SapModel: SAP2000 model
        average_values: if True, values in elements with the same x value are averaged
            (Set to False when comparing with SAP2000 interface tables)
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
    Output:
        results: dictionary containing information for the specific load case as follow:
            load_case: element_1 to element_N:
                element_1: P, M2 and M3 per each x coordinate within the element
    """
    # Select modal case
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput(load_case)
    raise_warning('Select modal case', ret)

    # Read information from SAP2000
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [NumResults, Element, Station, Mesh, MeshStation, LoadCase,
        StepType, StepNum, P, V2, V3, T, M2, M3, ret] = output
    raise_warning('Get element forces', ret)

    StepNum = list(StepNum)
    Element = list(Element)
    if round_coordinates:
        Station = list([outils.round_6_sign_digits(i) for i in Station])
        MeshStation = list([outils.round_6_sign_digits(i) for i in MeshStation])
    P, M2, M3 = list(P), list(M2), list(M3)

    # Save information into a dictionary
    StepNum_unique = list(set(StepNum))
    StepNum_unique.sort()
    Element_unique = outils.sort_list_string(list(set(Element)))

    results = dict()
    for mode in StepNum_unique:
        if mode > 0:
            continue
        mode_label = load_case
        results[mode_label] = dict()
        for element in Element_unique:
            element_label = 'Element_' + str(int(element))
            results[mode_label][element_label] = dict()

            # Bool list for for specific load case and element
            id_mode_element = [el == element and StepNum[i]
                               == mode and LoadCase[i] == load_case for i, el in enumerate(Element)]

            # Coordinates and forces retrievement
            station_mode_element = [st for i, st in enumerate(
                Station) if id_mode_element[i]]
            p = [force for i, force in enumerate(P) if id_mode_element[i]]
            m2 = [force for i, force in enumerate(M2) if id_mode_element[i]]
            m3 = [force for i, force in enumerate(M3) if id_mode_element[i]]

            if average_values:
                # values in elements with the same x value are averaged
                x = outils.sort_list_string(list(set(station_mode_element)))
                p_def, m2_def, m3_def = list(), list(), list()
                for i in x:
                    average_id = [i == x for x in station_mode_element]
                    p_aux = [p[j] for j, _ in enumerate(p) if average_id[j]]
                    m2_aux = [m2[j] for j, _ in enumerate(m2) if average_id[j]]
                    m3_aux = [m3[j] for j, _ in enumerate(m3) if average_id[j]]
                    p_def.append(np.mean(p_aux))
                    m2_def.append(np.mean(m2_aux))
                    m3_def.append(np.mean(m3_aux))
            else:
                # retrieve left-hand value in elements with the same x value (for validating purposes)
                x_coordinate_id = [st != station_mode_element[i-1] for i, st in enumerate(station_mode_element)]
                x = [x for i, x in enumerate(station_mode_element) if x_coordinate_id[i]]
                p_def = [force for i, force in enumerate(p) if x_coordinate_id[i]]
                m2_def = [force for i, force in enumerate(m2) if x_coordinate_id[i]]
                m3_def = [force for i, force in enumerate(m3) if x_coordinate_id[i]]

            mesh = [f'{element}.{mesh_id}' for mesh_id in range(len(x))]

            # Save into results dictionary
            results[mode_label][element_label]['x'] = x
            results[mode_label][element_label]['P'] = p_def
            results[mode_label][element_label]['M2'] = m2_def
            results[mode_label][element_label]['M3'] = m3_def
            results[mode_label][element_label]['Mesh_id'] = mesh

    return results


def get_elementsections(all_elements, all_elements_stat, SapModel):
    """
    Input:
        all_elements: list with all elements (e.g. ['1', '2'...])
        all_elements_stat: list with all elements stations (e.g. ['1-1', '1-2'...])
        Remark: both come from getnames_point_elements
    Return:
        dictionary giving section (e.g. "FSEC1") associated to each element
    Additional remarks:
        The only OAPI function found to get sections was LineElm.GetProperty
        (which takes element stations, not element, as inputs);
    """
    # Iteration over element
    element_section = dict()
    for element in outils.sort_list_string(all_elements):
        if element in all_elements_stat:  # it is not a station (non-meshed elements)
            start_str = element
            id_element = [start_str == element_stat for element_stat in all_elements_stat]
        else:  # it is a station (meshed elements)
            start_str = element + '-'
            id_element = [element_stat.startswith(
                start_str) for element_stat in all_elements_stat]
        elements_stat_element = [stat for i, stat in enumerate(
            all_elements_stat) if id_element[i]]
        aux_sections = list()
        # Get section for each sub-element of element
        for ElemName in elements_stat_element:

            PropName = ""
            ObjType = 0
            Var, sVarTotalLength, sVarRelStartLoc = 0, 0, 0

            output = SapModel.LineElm.GetProperty(
                ElemName, "", ObjType, Var, sVarTotalLength, sVarRelStartLoc)
            [PropName, ObjType, Var, sVarTotalLength, sVarRelStartLoc, ret] = output
            raise_warning("Get element's sections", ret)

            aux_sections.append(PropName)

        # If all sections are the same -> element has that section
        if len(set(aux_sections)) == 1:
            element_section[element] = aux_sections[0]
        else:  # e.g. "1-1 has "FSEC1" and "1-2" has "FSEC2"
            element_section[element] = dict()
            element_section[element]['stations'] = elements_stat_element
            element_section[element]['section'] = aux_sections
            message = 'More than one section assigned to an element'
            warnings.warn(message, UserWarning)

    return element_section


def get_section_information(all_sections, SapModel):
    """
    Function Duties:
        Gets geometry and material related to each section of all_sections
    """
    # A) Section properties
    section_properties = get_sectionproperties(all_sections, SapModel)

    # B) Material assigned in each section
    section_material = get_material_section(all_sections, SapModel)

    # C) Material properties
    all_materials = list(set([section_material[i]
                         for i in list(section_material)]))
    material_properties = get_material_properties(all_materials, SapModel)

    # D) Section properties and material (merge all)
    section_properties_material = outils.get_sectionproperties_material(section_properties,
                                                                        section_material,
                                                                        material_properties)
    return section_properties_material


def get_sectionproperties(all_sections, SapModel):
    """
    Function duties:
        Return a dictionary containing some section properties from a list of sections
    Remark: variables definition
        Area: cross-section (axial) area
        As2: Shear area in 2 direction
        As3: Shear area in 3 direction
        Torsion: Torsional Constant
        I22: Moment of Inertia about 2 axis
        I33: Moment of Inertia about 3 axis
        S22: Section modulus about 2 axis
        S33: Section modulus about 3 axis
        Z22: Plastic modulus about 2 axis
        Z33: Plastic modulus about 3 axis
        R22: Radious of Gyration about 2 axis
        R33: Radious of Gyration about 3 axis
    """
    section_properties = dict()
    for section in all_sections:
        Area, As2, As3, Torsion, I22, I33 = 0, 0, 0, 0, 0, 0
        S22, S33, Z22, Z33, R22, R33 = 0, 0, 0, 0, 0, 0
        output = SapModel.PropFrame.GetSectProps(
            section, Area, As2, As3, Torsion, I22, I33, S22, S33, Z22, Z33, R22, R33)

        [Area, As2, As3, Torsion, I22, I33, S22,
            S33, Z22, Z33, R22, R33, ret] = output
        raise_warning("Section properties", ret)
        section_properties[section] = {
            "Area": Area, "I22": I22, "I33": I33, "S22": S22, "S33": S33}

    return section_properties


def get_material_I_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is I section
    Input:
        SectionName: name of a section
    Return:
        is_I: bool (True if SectionName is I section)
        MatProp: name of the material (if section is I)
    """
    FileName, MatProp = "", ""
    t3, t2, tf, tw, t2b, tfb, Color = 0, 0, 0, 0, 0, 0, 0
    Notes, GUID = "", ""
    output = SapModel.PropFrame.GetISection(
        SectionName, FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_I = True
    else:
        is_I = False

    return is_I, MatProp


def get_material_rectangular_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is rectangular
    Input:
        SectionName: name of a section
    Return:
        is_rectangular: bool (True if SectionName is rectangular)
        MatProp: name of the material (if section is rectangular)
    """
    FileName, MatProp = "", ""
    t3, t2, Color = 0, 0, 0
    Notes, GUID = "", ""
    output = SapModel.PropFrame.GetRectangle(
        SectionName, FileName, MatProp, t3, t2, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_rectangular = True
    else:
        is_rectangular = False

    return is_rectangular, MatProp


def get_material_SD_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_SD: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    MyType: Main of them are (see OAPI doc for other types):
        1 = I-section
        2 = Channel
        3 = Tee
        4 = Angle
        5 = Double Angle
        6 = Box
        7 = Pipe
        8 = Plate
    DesignType:
        0 = No design
        1 = Design as general steel section
        2 = Design as a concrete column; check the reinforcing
        3 = Design as a concrete column; design the reinforcing
    """
    MatProp, ShapeName, Notes, GUID = "", "", "", ""
    NumberItems, Color, DesignType = 0, 0, 0
    MyType = []
    output = SapModel.PropFrame.GetSDSection(
        SectionName, MatProp, NumberItems, ShapeName, MyType, DesignType, Color, Notes, GUID)
    [MatProp, NumberItems, ShapeName, MyType,
        DesignType, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_SD = True
    else:
        is_SD = False

    return is_SD, MatProp


def get_material_angle_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_angle: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, t2, tf, tw = 0.0, 0.0, 0.0, 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetAngle(SectionName, FileName, MatProp, t3, t2, tf, tw,
                                         Color, Notes, GUID)
    [SectionName, MatProp, t3, t2, tf, tw, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_angle = True
    else:
        is_angle = False

    return is_angle, MatProp


def get_material_circle_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_circle: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp, t3 = "", "", 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetCircle(SectionName, FileName, MatProp, t3,
                                          Color, Notes, GUID)
    [FileName, MatProp, t3, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_circle = True
    else:
        is_circle = False

    return is_circle, MatProp


def get_material_tube_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_tube: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, t2, tf, tw = 0.0, 0.0, 0.0, 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetTube(SectionName, FileName, MatProp, t3, t2, tf,
                                        tw, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, tf, tw, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_tube = True
    else:
        is_tube = False

    return is_tube, MatProp


def get_material_general_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_general: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, t2, Area, As2, As3 = 0.0, 0.0, 0.0, 0.0, 0.0
    Torsion, I22, I33, S22, S33 = 0.0, 0.0, 0.0, 0.0, 0.0
    Z22, Z33, R22, R33 = 0.0, 0.0, 0.0, 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetGeneral(SectionName, FileName, MatProp, t3, t2,
                                           Area, As2, As3, Torsion, I22, I33, S22,
                                           S33, Z22, Z33, R22, R33, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, Area, As2, As3, Torsion, I22, I33,
     S22, S33, Z22, Z33, R22, R33, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_general = True
    else:
        is_general = False

    return is_general, MatProp


def get_material_pipe_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_pipe: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, tw = 0.0, 0.0  # Outside diameter and wall thickness
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetPipe(SectionName, FileName, MatProp, t3, tw, Color, Notes, GUID)

    [FileName, MatProp, t3, tw, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_pipe = True
    else:
        is_pipe = False

    return is_pipe, MatProp


def get_material_section(all_sections, SapModel):
    """
    Function Duties:
        Retrieves the material for each of the sections defined in all_sections
    Input:
        all_sections: list of section names
        SapModel: Sap model
    Return:
        section_material: dictionary containing the name of each material
            assigned to each section
    Remark:
        Only some sections have defined (I, SD, Rectangular...). Add more functions
            if other types are required (e.g. define get_material_channel_section)
            [for that, check all types of "Get" functions in OAPI by searching
            "MatProp"]
    """
    section_material = dict()

    for SectionName in all_sections:

        is_I, MatProp_I = get_material_I_section(SectionName, SapModel)
        is_SD, MatProp_SD = get_material_SD_section(SectionName, SapModel)
        is_rectangular, MatProp_rectangular = get_material_rectangular_section(
            SectionName, SapModel)
        is_angle, MatProp_angle = get_material_angle_section(SectionName, SapModel)
        is_circle, MatProp_circle = get_material_circle_section(SectionName, SapModel)
        is_tube, MatProp_tube = get_material_tube_section(SectionName, SapModel)
        is_general, MatProp_general = get_material_general_section(SectionName, SapModel)
        is_pipe, MatProp_pipe = get_material_pipe_section(SectionName, SapModel)
        # ... --> Define other functions if required (e.g. get_material_channel_section...)

        if is_I:
            MatProp = MatProp_I
        elif is_SD:
            MatProp = MatProp_SD
        elif is_rectangular:
            MatProp = MatProp_rectangular
        elif is_angle:
            MatProp = MatProp_angle
        elif is_circle:
            MatProp = MatProp_circle
        elif is_tube:
            MatProp = MatProp_tube
        elif is_general:
            MatProp = MatProp_general
        elif is_pipe:
            MatProp = MatProp_pipe
        # add other functions if required (e.g. for channel, etc. sections)
        else:
            MatProp = 'Not found'
            warnings.warn(
                f'No material found for {SectionName} section', UserWarning)

        section_material[SectionName] = MatProp

    return section_material


def get_modal_response_disp(input_data, SapModel):
    """
    Function Duties:
        Modifies sap model with data in input_data,
        runs analysis and return modal results with
        displacement mode shapes
    """
    # A) Unlock model and set IS units
    unlock_model(SapModel)
    set_ISunits(SapModel)

    # B) Set new material properties
    if 'material' in input_data:
        material_dict = input_data['material']
        set_materials(material_dict, SapModel)

    # C) Modify spring supports
    if 'joint_springs' in input_data:
        joint_dict = input_data['joint_springs']
        set_jointsprings(joint_dict, SapModel)

    # D) Modify partial fixity
    if 'frame_releases' in input_data:
        frame_releases_dict = input_data['frame_releases']
        set_framereleases(frame_releases_dict, SapModel)

    # E) Run Analysis
    run_analysis(SapModel)

    # F) Read results and save into a dictionary
    frequencies = get_modalfrequencies(SapModel)
    Name_points_group = "modeshape_points"
    disp_modeshapes = get_displmodeshapes(Name_points_group, SapModel)
    all_dict = {'frequencies': frequencies,
                'disp_modeshapes': disp_modeshapes}
    modal_results = outils.merge_all_dict(all_dict)

    return modal_results


def get_modal_response_strain(input_data, SapModel):
    """
    Function Duties:
        Modifies sap model with data in input_data,
        runs analysis and return modal results with
        strain mode shapes
    """
    # A) Unlock model and set IS units
    unlock_model(SapModel)
    set_ISunits(SapModel)

    # B) Set new material properties
    if 'material' in input_data:
        material_dict = input_data['material']
        set_materials(material_dict, SapModel)

    # C) Modify spring supports
    if 'joint_springs' in input_data:
        joint_dict = input_data['joint_springs']
        set_jointsprings(joint_dict, SapModel)

    # D) Modify partial fixity
    if 'frame_releases' in input_data:
        frame_releases_dict = input_data['frame_releases']
        set_framereleases(frame_releases_dict, SapModel)

    # E) Run Analysis
    run_analysis(SapModel)

    # F) Read results and save into a dictionary
    frequencies = get_modalfrequencies(SapModel)

    Name_points_group, Name_elements_group = "allpoints", "allframes"
    all_points, all_elements, all_elements_stat = getnames_point_elements(Name_points_group,
                                                                                Name_elements_group,
                                                                                SapModel)
    element_section = get_elementsections(all_elements, all_elements_stat, SapModel)
    all_sections = list(set([element_section[i] for i in list(element_section)]))
    section_properties_material = get_section_information(all_sections, SapModel)
    # Modal Forces and strain modeshapes
    Name_elements_group = 'modeshape_frames'
    modal_forces = get_modalforces(Name_elements_group, SapModel)
    strain_modeshapes = outils.get_strainmodeshapes(modal_forces, element_section, section_properties_material)

    # Save results
    all_dict = {'frequencies': frequencies,
                'strain_modeshapes': strain_modeshapes}
    modal_results = outils.merge_all_dict(all_dict)

    return modal_results


def get_modal_response(input_data, SapModel, get_disp=True, get_strain=True):
    """
    Function Duties:
        Modifies sap model with data in input_data,
        runs analysis and return modal results with
        strain mode shapes
    """
    # A) Unlock model and set IS units
    unlock_model(SapModel)
    set_ISunits(SapModel)

    # B) Set new material properties
    if 'material' in input_data:
        material_dict = input_data['material']
        set_materials(material_dict, SapModel)

    # C) Modify spring supports
    if 'joint_springs' in input_data:
        joint_dict = input_data['joint_springs']
        set_jointsprings(joint_dict, SapModel)

    # D) Modify partial fixity
    if 'frame_releases' in input_data:
        frame_releases_dict = input_data['frame_releases']
        set_framereleases(frame_releases_dict, SapModel)

    # E) Run Analysis
    run_analysis(SapModel)

    # F) Read results and save into a dictionary
    frequencies = get_modalfrequencies(SapModel)

    if get_disp:
        Name_points_group = "modeshape_points"
        disp_modeshapes = get_displmodeshapes(Name_points_group, SapModel)
    else:
        disp_modeshapes = None

    if get_strain:
        Name_points_group, Name_elements_group = "allpoints", "allframes"
        all_points, all_elements, all_elements_stat = getnames_point_elements(Name_points_group,
                                                                                    Name_elements_group,
                                                                                    SapModel)
        element_section = get_elementsections(all_elements, all_elements_stat, SapModel)
        all_sections = list(set([element_section[i] for i in list(element_section)]))
        section_properties_material = get_section_information(all_sections, SapModel)
        # Modal Forces and strain modeshapes
        Name_elements_group = 'modeshape_frames'
        modal_forces = get_modalforces(Name_elements_group, SapModel)
        strain_modeshapes = outils.get_strainmodeshapes(modal_forces, element_section, section_properties_material)
    else:
        strain_modeshapes = None

    # Save results
    all_dict = {'frequencies': frequencies,
                'disp_modeshapes': disp_modeshapes,
                'strain_modeshapes': strain_modeshapes}
    modal_results = outils.merge_all_dict(all_dict)

    return modal_results


def get_pointrestraints(point_list, SapModel):
    """
    Input:
        List of points (point_list[i] = str)
    Output:
        Dictionary containing a list of boolean values for
        each point.
        Remark:
            len(restraints[i]) = num_dof
            restraints[i][j] = True if there is restriction in dof j
                for point i
    """
    restraints = dict()
    for i in point_list:
        Value = []
        Value, ret = SapModel.PointObj.GetRestraint(str(i), Value)
        restraints[i] = list(Value)

    raise_warning("Get joint restraints", ret)

    return restraints


def plot_accelerometers_as_forces(Phi_id, n_modes_considered, SapModel) -> None:
    """
    Function Duties:
        Plot sensors locations (given by Phi_id) as forces
        in the SAP2000 model
    Input:
        Phi_id: list of strings with the joint names followed by
            the DOF (e.g. ['1_U1', '3_U2'])
        n_modes_considered: number of modes aimed to find with EfI
        SapModel: SAP2000 model
    Remark:
        After executing the function, navigate through SAP2000 interface
        and click on:
        Display -> Show Object Load Assigns -> Joint ->
            -> "OSP_{n_sensors}_sensors" Load Case
    """
    # Initial variables
    ref_force = 10**5  # magnitude of the plotted force
    # always like that (SAP2000)
    all_dofs = ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']
    n_sensors = len(Phi_id)

    # A) Unlock model
    unlock_model(SapModel, lock=False)

    # B) Define load pattern
    LoadPatName = f"OSP_{n_modes_considered}_modes_{n_sensors}_ACCELEROMETERS"

    # if the load pattern already exists, delete it:
    ret = SapModel.LoadPatterns.Delete(LoadPatName)

    MyType = 3  # 3: Live
    ret = SapModel.LoadPatterns.Add(LoadPatName, MyType, 0, False)
    raise_warning("Create load pattern", ret)

    # C) Plot forces
    # C.1: regrouping Phi_id
    joint_dict = dict()
    for i in Phi_id:
        joint = i.split('_')[0]
        dof = i.split('_')[1]
        if joint not in joint_dict:
            joint_dict[joint] = list()
        joint_dict[joint].append(dof)

    # C.2: plotting forces
    for joint in list(joint_dict):
        dof = joint_dict[joint]
        active = [i in dof for i in all_dofs]
        Value = np.array([0, 0, 0, 0, 0, 0])
        Value[active] = ref_force
        Value = list(Value)
        _, ret = SapModel.PointObj.SetLoadForce(
            joint, LoadPatName, Value, False)
        raise_warning(f"Add load for joint: {joint}", ret)

    # Refresh view for all windows
    ret = SapModel.View.RefreshView()
    raise_warning("Refresh view", ret)


def plot_SG_as_forces(Psi_id, points_for_plotting, n_modes_considered, SapModel) -> None:
    """
    Function Duties:
        Plot sensors locations (given by Phi_id) as forces
        in the SAP2000 model
    Input:
        Psi_id: list of strings with the element names with the station
            followed by the location (e.g. ['2.0_up', '6.1_right'])
        points_for_plotting: dictionary coming from prepare_dict_plotting_SG
        n_modes_considered: number of modes aimed to find with EfI
        SapModel: SAP2000 model
    Remark:
        After executing the function, navigate through SAP2000 interface
        and click on:
        Display -> Show Object Load Assigns -> Joint ->
            -> "OSP_{n_sensors}_sensors" Load Case
    """
    # Initial variables
    ref_force = 10**5  # magnitude of the plotted force
    n_sensors = len(Psi_id)

    # A) Unlock model
    unlock_model(SapModel, lock=False)

    # B) Define load pattern
    LoadPatName = f"OSP_{n_modes_considered}_modes_{n_sensors}_SG"

    # if the load pattern already exists, delete it:
    ret = SapModel.LoadPatterns.Delete(LoadPatName)

    MyType = 3  # 3: Live
    ret = SapModel.LoadPatterns.Add(LoadPatName, MyType, 0, False)

    LoadPat = LoadPatName
    MyType = 1  # 1: Force
    CSys = 'Local'
    RelDist, Replace = False, False
    ItemType = 0  # 0: Object
    for element in list(points_for_plotting):
        Name = element.split('_')[1]
        for Dist, location in zip(points_for_plotting[element]['x'], points_for_plotting[element]['location']):
            if location in ('up', 'down'):
                Dir = 2
                Sense = -1 if location == 'up' else 1
            elif location in ('left', 'right'):
                Dir = 3
                Sense = -1 if location == 'right' else 1
            Val = ref_force * Sense
            ret = SapModel.FrameObj.SetLoadPoint(Name, LoadPat, MyType, Dir, Dist, Val, CSys, RelDist, Replace, ItemType)
            raise_warning(f"Plotting {element} at distance {Dist}", ret)

    # Refresh view for all windows
    ret = SapModel.View.RefreshView()
    raise_warning("Refresh view", ret)


# def old_get_material_I_section(all_sections, SapModel):
#     """
#     Function duties:
#         Gets the material for each of sections given in all_sections
#     Input:
#         all_sections: list of sections
#     Remark:
#         Function to be used with I sections (for other sections, i.e. rectangle)
#         use another sap function
#     """
#     section_material = dict()
#     for SectionName in all_sections:
#         FileName, MatProp = "", ""
#         t3, t2, tf, tw, t2b, tfb, Color = 0, 0, 0, 0, 0, 0, 0
#         Notes, GUID = "", ""
#         output = SapModel.PropFrame.GetISection(SectionName, FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID)
#         [FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID, ret] = output
#         raise_warning(f'Get material from {SectionName} section', ret)

#         section_material[SectionName] = MatProp

#     return section_material


def get_material_properties(all_materials, SapModel):
    """
    Function duties:
        Gets material properties from a list of materials
    Remark:
        e: Modulus of Elasticity, E
        u: Poisson
        a: Coefficient of Thermal Expansion
        g: Shear modulus
    """
    material_dict = dict()

    for MatName in all_materials:
        e, u, a, g = 0, 0, 0, 0
        output = SapModel.PropMaterial.GetMPIsotropic(MatName, e, u, a, g)
        [e, u, a, g, ret] = output
        raise_warning("Get material properties", ret)

        material_dict[MatName] = {"E": e, "u": u, "a": a}

    return material_dict


def set_solver(SapModel, SolverType=2, SolverProcessType=2,
               NumberParallelRuns=0, ResponseFileSizeMaxMB=-1, NumberAnalysisThreads=-1,
               StiffCase="MODAL") -> None:
    """
    Analysis options:
    SolverType:
        0 = Standard solver
        1 = Advanced solver
        2 = Multi-threaded solver
    SolverProcessType:
        0 = Auto (program determined)
        1 = GUI process
        2 = Separate process (out of GUI)
    NumberParallelRuns: integer between -8 and 8, inclusive, not including -1
        -8 to -2 = The negative of the program determined value when the assigned value is
        0 = Auto parallel (use up to all physical cores - max 8).
        1 = Serial.
        2 to 8 = User defined parallel (use up to this fixed number of cores - max 8).
    ResponseFileSizeMaxMB: The maximum size of a response file in MB before a new response file
        is created. Positive if user specified, negative if program determined.
    NumberAnalysisThreads: Number of threads that the analysis can use. Positive if user specified,
        negative if program determined.
    StiffCase: The name of the load case used when outputting the mass and stiffness matrices to
        text files If this item is blank, no matrices are output.
    """
    # Set the solver options
    ret = SapModel.Analyze.SetSolverOption_3(
        SolverType, SolverProcessType, NumberParallelRuns, ResponseFileSizeMaxMB, NumberAnalysisThreads, StiffCase)
    raise_warning("Set solver options", ret)
