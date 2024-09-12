
import matplotlib.pyplot as plt
import os

import helpers.outils as outils
import helpers.classes as classes

"""
File used to get data from arduino recordments (in DOCTORADO_CODES/OMA) to be used in this project)

- Correct timestamps
- Filter Outliers
- Save in a csv with appropiate headers
- Save metadata in a json file
"""
# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
# File names:
"""
There must be always one recording in rawdata_strain paht.
Additionally, it will be checked if there is a recording in rawdata_acc path
    called test_name + '_acc.txt'
"""
test_name = 'test14'
latex_style = False
sg_info_name = 'sg_channels_info.csv'
acc_info_name = 'acc_channels_info.csv'
project_name = 'EGM STEEL FRAME - OMA BY STRAIN GAUGES'  # for UFF file
num_plotted = 6  # number of channels plotted in the report figure (multiple of 9 is recommended)

# Input and output paths
username = outils.get_username()
config = outils.read_config_file(os.path.join(
    'src', 'signals_for_oma', 'paths', username + '.txt'))
rawdata_strain = config['rawdata_strain']
rawdata_acc = config['rawdata_acc']
savedata = config['savedata']
savefigures = config['savefigures']

# Check if acceleration data exists
exist_acc = False
if test_name + '_acc.txt' in os.listdir(rawdata_acc):
    exist_acc = True

# Latex style
classes.use_latex_style(latex_style)
# ---------------------------------------------------------------

if exist_acc:
    test_name_all = [test_name, test_name + '_acc']
    sensor_info_name = [sg_info_name, acc_info_name]
    rawdata_all = [rawdata_strain, rawdata_acc]
    data_type_all = ['Strain', 'Acceleration']
    units_all = ['m/m', 'm/s2']
else:
    test_name_all, sensor_info_name, rawdata_all = [
        test_name], [sg_info_name], [rawdata_strain]
    data_type_all, units_all = ['Strain'], ['m/m']
parameters_name_all = [i + '_metadata' for i in test_name_all]

for test_name, parameters_name, sensor_info_name, rawdata, data_type, units in zip(test_name_all, parameters_name_all,
                                                                                   sensor_info_name, rawdata_all,
                                                                                   data_type_all, units_all):
    # Signal Processing
    processor = classes.SignalProcessing(test_name, parameters_name, sensor_info_name,
                                         rawdata, savedata, savefigures)
    processor.check_files_existence()
    processor.create_folders()
    processor.read_raw_files()
    # fig_all, ax_all = processor.plot_raw_data()
    plt.close('all')
    processor.filter_signal()
    if processor.input_parameters['apply_outliers']:
        fig_all, ax_all = processor.plot_outliers_results()
    plt.close('all')
    fig_all, ax_all = processor.plot_filter_results()
    plt.close('all')
    fig, ax = processor.plot_raw_data_report(num_plotted=num_plotted)
    if data_type == 'Strain':
        processor.get_strain_data()
    else:
        processor.get_acc_data(already_detrended=True)
    processor.save_measurement_data()
    processor.save_test_info_summary(data_type=data_type, units=units)

    # Fast PSD and SVD
    analysis = classes.SignalAnalysis(test_name, savefigures, savedata)
    analysis.compute_psd_svd(pov=0.5, df=0.01)
    fig, ax = analysis.plot_psd_matrix(fmax=40)
    analysis.get_channel_power()
    fig, ax = analysis.plot_sv(fmax=35)
    plt.close('all')

    # UFF Generation
    uff_generator = classes.UFFGenerator(
        test_name, project_name, savedata, savedata)
    uff_generator.check_files_existence()
    uff_generator.read_files()
    uff_generator.remove_existing_uff_file()
    uff_generator.generate_uff_data()
