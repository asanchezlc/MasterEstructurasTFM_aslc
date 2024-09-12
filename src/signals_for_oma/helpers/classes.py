
import copy
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyuff

from scipy import signal

import helpers.outils_arduino as outils_arduino
import helpers.outils as outils


def use_latex_style(bool=True) -> None:
    if bool:
        # Figures configuration
        mpl.use("pgf")
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",  # Change this if using xelatex or lualatex
            "font.family": "serif",  # Use a more commonly available font family
            "mathtext.fontset": "stix",  # Use STIX for math text
            "font.size": 17,  # General font size
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,  # Specific font size for x-tick labels
            "ytick.labelsize": 15,  # Specific font size for y-tick labels
            "legend.fontsize": 15,
            "figure.titlesize": 17,
            "text.usetex": True,  # Use LaTeX to write all text
            "pgf.rcfonts": False,  # Don't setup fonts from rc parameters
            # Use a raw string for the preamble
            "pgf.preamble": r"\usepackage{amsmath}",
        })


class SignalProcessing:
    """
    Class Information:
        Class used to process raw data coming from SerialPlot (Arduino) from SG
        measurements and convert it into strain data
    Attributes:
        test_name (str): Name of the test.
        parameters_name (str): Name of the parameters file (contains information about both
            acquisition data and filtering parameters).
        sensor_info_name (str): Name of the file with the sensor coordinates information.
        rawdata_path (str): Path where the raw data is stored.
        savedata_path (str): Path where the processed data will be saved.
        savefig_path (str): Path where the figures will be saved.
    Class Methods that save files:
        plot_outliers_results: save figures about outliers filtering
        plot_filter_results: save figures about general filtering
        save_measurement_data: Save the processed strain data
        save_test_info_summary: Save the processed test information
    Remarks:
        Filtering parameters are chosen in a iterative process. plot_raw_data method
        allows us to check the raw data and adjust the filtering parameters, and
        plot_outliers_results and plot_filter_results allow us to check the results
    """

    def __init__(self, test_name: str, parameters_name: str, sensor_info_name: str,
                 rawdata_path: str, savedata_path: str, savefig_path: str):
        # Class parameters
        self.test_name = test_name
        self.parameters_name = parameters_name
        self.sensor_info_name = sensor_info_name
        self.rawdata_path = rawdata_path
        self.savedata_path = savedata_path
        self.savefig_path = savefig_path
        # Input files
        self.data_name = test_name + '.txt'
        self.metadata_name = parameters_name + '.txt'
        # Output files
        self.data_name_output = test_name + '.csv'
        self.test_info_name_output = test_name + '_information.json'
        # Class variables
        self.df_raw = None
        self.input_parameters = None
        # Signal processing variables
        self.fs = None
        self.time_raw = None
        self.time = None
        # Filter variables
        self.df_raw_cut = None
        self.df_outliers = None
        self.df_filtered = None
        self.peaks_all = None  # Outliers detected in each channel
        # physical magnitude (e.g. acceleration or strain)
        self.df_measurement = None

    def check_files_existence(self):
        """Check if both test and parameters information are in rawdata path"""
        required_files = [self.data_name,
                          self.metadata_name, self.sensor_info_name]
        for file in required_files:
            if file not in os.listdir(self.rawdata_path):
                raise FileNotFoundError(
                    f"File {file} not found in {self.rawdata_path}")

    def create_folders(self):
        """Creates folders for saving results"""
        if not os.path.exists(self.savedata_path):  # for saving data
            os.makedirs(self.savedata_path)
        # for saving figures
        if self.test_name not in os.listdir(self.savefig_path):
            os.makedirs(os.path.join(self.savefig_path, self.test_name))

    def read_raw_files(self) -> None:
        """
        Read raw data and metadata files and save them as class variables
        Remark:
            SerialPlot does not properly save time decimals
        """
        input_parameters = outils.read_config_file(
            os.path.join(self.rawdata_path, self.metadata_name))

        if 'simulated_oma' not in input_parameters:
            input_parameters['simulated_oma'] = 0
        input_parameters['simulated_oma'] = bool(input_parameters['simulated_oma'])

        if input_parameters['simulated_oma']:
            df_raw = pd.read_csv(os.path.join(
                self.rawdata_path, self.data_name))
        else:
            df_raw = pd.read_csv(os.path.join(
                self.rawdata_path, self.data_name), dtype=str)

        for col in df_raw.columns:
            if isinstance(df_raw[col][0], str):
                df_raw[col] = outils_arduino.read_decimals(df_raw[col].values)

        # Adjust time and sampling frequency
        df_raw['timestamp'] = df_raw['timestamp'] - df_raw['timestamp'].iloc[0]

        # Adjust boolean variables
        input_parameters['apply_detrend'] = bool(
            input_parameters['apply_detrend'])
        input_parameters['apply_butterworth'] = bool(
            input_parameters['apply_butterworth'])
        input_parameters['apply_decimate'] = bool(
            input_parameters['apply_decimate'])
        input_parameters['apply_outliers'] = bool(
            input_parameters['apply_outliers'])

        # Save class variables
        self.time_raw = df_raw['timestamp'].values
        self.fs = 1/np.diff(self.time_raw).mean()
        self.input_parameters = input_parameters
        self.df_raw = df_raw

    def plot_raw_data(self):
        """
        Plot raw data in order to configure the filtering parameters
        This plot is not saved
        """
        fig_all, ax_all = list(), list()
        for channel in list(self.df_raw.drop(columns=['timestamp'])):
            fig, ax = plt.subplots()
            ax.plot(self.time_raw, self.df_raw[channel])
            ax.set_title(f'Raw data; {channel}')
            ax.grid()
            ax.set_xlabel('Time [s]')
            if self.input_parameters['simulated_oma']:
                ax.set_ylabel('Strain')
            else:
                ax.set_ylabel('Bits')
            fig_all.append(fig)
            ax_all.append(ax)

        return fig_all, ax_all

    def filter_signal(self) -> None:
        """Signal processing based on the input parameters"""
        df_raw = copy.deepcopy(self.df_raw)

        # Delete wrong channels
        if self.input_parameters['ch_to_delete']:
            deleted_channels = ['Channel ' + str(i)
                                for i in self.input_parameters['ch_to_delete']]
            df_raw = df_raw.drop(columns=deleted_channels)

        # Delete first part of recording (if specified)
        df_raw = df_raw.iloc[self.input_parameters['timestart_id']:]
        df_raw = df_raw.reset_index(drop=True)

        # Delete last part of recording (if specified)
        if 'timeend_id' in list(self.input_parameters):
            if self.input_parameters['timeend_id'] and self.input_parameters['timestart_id']:
                if self.input_parameters['timeend_id'] > self.input_parameters['timestart_id']:
                    df_raw = df_raw.iloc[:self.input_parameters['timeend_id']]
                    df_raw = df_raw.reset_index(drop=True)

        # Apply outliers filter
        df_outliers = copy.deepcopy(df_raw)
        if self.input_parameters['apply_outliers']:
            peaks_all = dict()
            for channel in list(df_raw.drop(columns=['timestamp'])):
                # Manual Filter
                df_outliers[channel], peaks_all_manual = outils.manual_filter(
                    df_outliers[channel], self.input_parameters['distance'])

                # Peak Picking Filter
                peaks_all_pp = np.array([])
                for i in range(self.input_parameters['outliers_order']):
                    m = copy.deepcopy(df_outliers[channel])
                    peaks = outils.peaks_pos_neg(
                        m, threshold=self.input_parameters['threshold'],
                        prominence=self.input_parameters['prominence'])
                    if len(peaks) == 0:
                        continue
                    peaks_all_pp = np.hstack((peaks_all_pp, peaks))
                    df_outliers[channel] = outils.interpolate_peaks(
                        df_outliers[channel], peaks)

                # Retrieve all filtered peaks
                peaks_all_ch = np.hstack((peaks_all_manual, peaks_all_pp))
                peaks_all[channel] = peaks_all_ch

        # Apply butter filter
        df_butter = copy.deepcopy(df_outliers)
        if self.input_parameters['apply_butterworth']:
            for channel in list(df_butter.drop(columns=['timestamp'])):
                df_butter[channel] = outils.butterworth_filter(
                    df_butter[channel], self.input_parameters['f_low'],
                    self.input_parameters['f_high'], self.fs, self.input_parameters['filter_order'])

        # Detrend
        df_detrended = copy.deepcopy(df_butter)
        if self.input_parameters['apply_detrend']:
            for channel in list(df_detrended.drop(columns=['timestamp'])):
                df_detrended[channel] = signal.detrend(df_detrended[channel])

        # Decimate
        df_decimated = copy.deepcopy(df_detrended)
        if self.input_parameters['apply_decimate']:
            df_decimated = pd.DataFrame(columns=df_detrended.columns)
            for channel in list(df_decimated.drop(columns=['timestamp'])):
                df_decimated[channel] = signal.decimate(
                    df_decimated[channel], self.input_parameters['decimation_factor'])
            fs = self.fs/self.input_parameters['decimation_factor']
            df_decimated['timestamp'] = list(
                np.arange(0, len(df_decimated['timestamp'])/fs, 1/fs))
            self.fs = fs

        # Save results
        self.time = df_decimated['timestamp']
        self.df_raw_cut = df_raw
        if self.input_parameters['apply_outliers']:
            self.peaks_all = peaks_all
        self.df_outliers = copy.deepcopy(df_outliers)
        self.df_filtered = copy.deepcopy(df_decimated)

    def plot_outliers_results(self):
        """Plot and save outliers results"""
        df_raw_cut = self.df_raw_cut
        df_outliers = self.df_outliers
        if df_outliers is None:
            raise ValueError('Outliers results are not available')
        fig_all, ax_all = list(), list()
        for channel in list(df_outliers.drop(columns=['timestamp'])):
            peaks_all_ch = self.peaks_all[channel]
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(self.time, df_raw_cut[channel])
            ax[0].plot(self.time[peaks_all_ch],
                       df_raw_cut[channel][peaks_all_ch], "x")
            ax[0].set_title('Raw data')
            ax[0].grid()
            ax[0].set_ylabel('Bits')
            ax[1].plot(self.time, df_outliers[channel])
            ax[1].set_title('Filtered data')
            ax[1].grid()
            ax[1].set_xlabel('Time [s]')
            ax[1].set_ylabel('Bits')
            fig.suptitle(f'Raw data and detected outliers for {channel}')
            fig.tight_layout()
            figname = 'Outliers Filter - ' + channel
            fig.savefig(os.path.join(
                self.savefig_path, self.test_name, figname))
            print(
                f"Percentage of removed peaks in {channel}:': {format(100*len(peaks_all_ch)/len(self.time), '.2f')}%")
            fig_all.append(fig)
            ax_all.append(ax)

        print(
            f'Outliers figures saved in {self.savefig_path}/{self.test_name}')
        return fig_all, ax_all

    def plot_filter_results(self):
        """Plot and save general filtering results"""
        df_outliers = self.df_outliers
        df_filtered = self.df_filtered
        if df_filtered is None:
            raise ValueError('Filter results are not available')
        fig_all, ax_all = list(), list()
        for channel in list(df_filtered.drop(columns=['timestamp'])):
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(self.time, df_outliers[channel] -
                       np.mean(df_outliers[channel]), label='Raw')
            ax[0].set_title('Raw data (with zero-mean)')
            ax[0].grid()
            if self.input_parameters['simulated_oma']:
                ax[0].set_ylabel('Strain')
            else:
                ax[0].set_ylabel('Bits')
            ax[0].set_ylim([1.1*np.min(df_filtered[channel]),
                            1.1*np.max(df_filtered[channel])])
            ax[1].plot(self.time, df_filtered[channel], label='Filtered')
            ax[1].set_title('Filtered data')
            ax[1].grid()
            ax[1].set_xlabel('Time [s]')
            if self.input_parameters['simulated_oma']:
                ax[1].set_ylabel('Strain')
            else:
                ax[1].set_ylabel('Bits')
            ax[1].set_ylim([1.1*np.min(df_filtered[channel]),
                            1.1*np.max(df_filtered[channel])])
            fig.suptitle(f'Raw data and Filtered data for {channel}')
            fig.tight_layout()
            figname = 'General Filter  - ' + channel
            fig.savefig(os.path.join(
                self.savefig_path, self.test_name, figname))
            fig_all.append(fig)
            ax_all.append(ax)

        print(f'Filter figures saved in {self.savefig_path}/{self.test_name}')
        return fig_all, ax_all

    def plot_raw_data_report(self, num_plotted=6):
        """Plot and save num_plotted first channels"""
        df_outliers = self.df_outliers
        df_filtered = self.df_filtered
        if df_filtered is None:
            raise ValueError('Filter results are not available')
        fig, ax = plt.subplots(num_plotted//3, num_plotted//2, figsize=(12, 12//(num_plotted//3)))
        y_1, y_2 = 0, 0
        for i, channel in enumerate(list(df_filtered.drop(columns=['timestamp']))[0:num_plotted]):
            row, col = i // 3, i % 3
            ax[row, col].plot(self.time, df_outliers[channel] -
                              np.mean(df_outliers[channel]), label='Raw')
            y_1 = min(1.1*np.min(df_outliers[channel]-np.mean(df_outliers[channel])), y_1)
            y_2 = max(1.1*np.max(df_outliers[channel]-np.mean(df_outliers[channel])), y_2)
            ax[row, col].grid(True, ls=":")
            ax[row, col].set_title(channel)
            if col == 0:
                if self.input_parameters['simulated_oma']:
                    ax[row, col].set_ylabel('Strain')
                else:
                    ax[row, col].set_ylabel('Bits')
            if row == num_plotted//3-1:
                ax[row, col].set_xlabel('Time [s]')
        for i, _ in enumerate(list(df_filtered.drop(columns=['timestamp']))[0:num_plotted]):
            row, col = i // 3, i % 3
            ax[row, col].set_ylim([y_1, y_2])
        fig.tight_layout()

        figname = 'Raw Data Report'
        fig.savefig(os.path.join(
            self.savefig_path, self.test_name, figname))

        print(f'Raw Data Report figure saved in {self.savefig_path}/{self.test_name}')

        return fig, ax

    def get_strain_data(self) -> None:
        """Convert filtered data (bits) into strain data"""
        # Convert to strain
        df_strain = pd.DataFrame(columns=self.df_filtered.columns)
        # Corrected time (if decimation and equal time steps)
        df_strain['timestamp'] = list(
            np.arange(0, len(self.df_filtered['timestamp'])/self.fs, 1/self.fs))

        if self.input_parameters['simulated_oma']:
            col_names = [col for col in self.df_filtered.columns if col != 'timestamp']
            df_strain[col_names] = copy.deepcopy(self.df_filtered[col_names])
        else:
            for ch in list(self.df_filtered.drop(columns=['timestamp'])):
                v0 = outils_arduino.DAC(self.df_filtered[ch], n_bits=self.input_parameters['n_bits'],
                                        Vs=self.input_parameters['Vs'], gain=self.input_parameters['GAIN'])
                strain = outils_arduino.V0_2_strain(
                    v0, Vs=self.input_parameters['Vs'], GF=self.input_parameters['GF'])
                df_strain[ch] = strain

        self.df_measurement = df_strain

    def get_acc_data(self, already_detrended=False) -> None:
        """Convert filtered data (bits) into acc data"""
        # Convert to acc
        df_acc = pd.DataFrame(columns=self.df_filtered.columns)
        # Corrected time (if decimation and equal time steps)
        df_acc['timestamp'] = list(
            np.arange(0, len(self.df_filtered['timestamp'])/self.fs, 1/self.fs))

        for ch in list(self.df_filtered.drop(columns=['timestamp'])):
            acc = outils_arduino.from_bits_2_acc(self.df_filtered[ch], n_bits=self.input_parameters['n_bits'],
                                                 Vs=self.input_parameters['Vs'], sensitivity=self.input_parameters['sensitivity'],
                                                 centered_V=self.input_parameters['centered_V'], already_detrended=already_detrended)
            df_acc[ch] = acc

        self.df_measurement = df_acc

    def save_measurement_data(self) -> None:
        """Save strains/accelerations filtered data into a csv"""
        df_data = copy.deepcopy(self.df_measurement)
        df_data.rename(columns={'timestamp': 't'}, inplace=True)
        df_data.rename(columns={channel: 'channel_' + channel.split(
            ' ')[1] for channel in list(df_data.drop(columns=['t']))}, inplace=True)
        df_data.to_csv(os.path.join(self.savedata_path,
                                    self.data_name_output), index=False, sep=';')
        print(
            f'Processed data saved as {self.data_name_output} in {self.savedata_path}')

    def save_test_info_summary(self, data_type='Strain', units='m/m') -> None:
        """Save test information summary into a json file"""
        df_metadata = pd.read_csv(os.path.join(
            self.rawdata_path, self.sensor_info_name), delimiter=';', comment='#')
        df_metadata['Channel'] = df_metadata['Channel'].astype(int)

        date = outils.get_file_modification_date(
            os.path.join(self.rawdata_path, self.data_name))

        directions = dict()
        coordinates = dict()

        for i in df_metadata['Channel']:
            channel_label = 'Channel ' + str(i)  # name assigned by serialplot
            if channel_label not in list(self.df_measurement):
                continue
            if self.input_parameters['ch_to_delete']:
                if i in self.input_parameters['ch_to_delete']:
                    continue
            if data_type == 'Strain':
                directions['channel_' + str(i)] = 'z'
            else:
                directions['channel_' + str(
                    i)] = df_metadata.loc[df_metadata['Channel'] == i, 'SAP2000_dir'].iloc[0]
            coordinates['channel_' + str(i)] = {'x': df_metadata.loc[df_metadata['Channel'] == i, 'x'].iloc[0],
                                                'y': df_metadata.loc[df_metadata['Channel'] == i, 'y'].iloc[0],
                                                'z': df_metadata.loc[df_metadata['Channel'] == i, 'z'].iloc[0]}
        signal_processing = copy.deepcopy(self.input_parameters)
        damaged_piles = None
        if 'damaged_piles' in self.input_parameters:
            damaged_piles = copy.deepcopy(
                self.input_parameters['damaged_piles'])
            signal_processing.pop('damaged_piles')

        metadata = dict()
        if data_type == 'Acceleration':
            metadata[
                'IMPORTANT'] = f'Further information about the test can be found in the sg test information ({self.test_name.replace("_acc", "")}_information.json)'
        metadata['fs'] = self.fs
        metadata['data_type'] = data_type
        metadata['units'] = units
        metadata['directions'] = directions
        # by default (to be used when comparing against SAP2000)
        if data_type == 'Strain':
            metadata['dir'] = {ch: 'up' for ch in directions}
        else:
            metadata['dir'] = {ch: None for ch in directions}
        metadata['coordinates'] = coordinates
        if damaged_piles:
            metadata['damaged_piles'] = damaged_piles
        metadata['signal_processing'] = signal_processing
        metadata['test_date'] = date

        with open(os.path.join(self.savedata_path, self.test_info_name_output), 'w') as f:
            json.dump(metadata, f)

        print(
            f'Processed test info saved as {self.test_info_name_output} in {self.savedata_path}')


class SignalAnalysis:
    """
    Class used to perform signal analysis on processed strain data and generate plots.

    Attributes:
        test_name (str): Name of the test.
        savepath_figures (str): Path where the figures will be saved.
        data_path (str): Path where the processed data is stored.
        n_sv (int): Number of singular values to plot.
    """

    def __init__(self, test_name: str, savepath_figures: str, data_path: str, n_sv: int = 5):
        self.test_name = test_name
        self.savepath_figures = savepath_figures
        self.data_path = data_path
        self.n_sv = n_sv
        self.data = None
        self.f = None
        self.PSD_matr = None
        self.S_val = None
        self.channel_power = None

        self._create_directories()
        self._read_data()

    def _create_directories(self) -> None:
        """Creates directories for saving figures."""
        if self.test_name not in os.listdir(self.savepath_figures):
            os.makedirs(os.path.join(self.savepath_figures, self.test_name))

    def _read_data(self) -> None:
        """Reads processed data from CSV file."""
        self.data = pd.read_csv(os.path.join(
            self.data_path, f'{self.test_name}.csv'), sep=';')

    def compute_psd_svd(self, pov: float = 0.5, df: float = 0.01) -> None:
        """Computes the PSD and SVD of the data."""
        (self.f, self.PSD_matr), (self.S_val, _) = outils.get_PSD_SVD(
            self.data, pov=pov, df=df)

    def plot_psd_matrix(self, fmax=None):
        """Plots and saves the PSD matrix (diagonal terms)."""
        channels = list(self.data.drop(columns=['t']))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, ch in enumerate(channels):
            ax.plot(self.f, self.PSD_matr[i, i, :], label=ch)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD')
        ax.set_title('PSD Matrix - Diagonal terms')
        ax.set_yscale('log')
        ax.legend()
        ax.grid()
        if fmax:
            ax.set_xlim([0, fmax])
        fig.tight_layout()
        figname = 'PSD_Matrix'
        fig.savefig(os.path.join(
            self.savepath_figures, self.test_name, figname))
        print(
            f'PSD Matrix figure saved as {figname} in {os.path.join(self.savepath_figures, self.test_name)}')
        return fig, ax

    def get_channel_power(self) -> None:
        """
        Retrieves the total power (integral over of PSD diagonal elements) for
        each channel, and print it
        """
        channels = list(self.data.drop(columns=['t']))
        power_all = dict()
        for i in range(np.shape(self.PSD_matr)[0]):
            power = np.sum(self.PSD_matr[i, i, :] * np.diff(self.f)[0])
            power_all[channels[i]] = np.real(power)
        id_descending = np.argsort(-np.array([power_all[i] for i in channels]))
        self.channel_power = power_all

        # Print results
        print('-----------------------------------------------------')
        print('--  Channels sorted from highest to lowest energy  --')
        print('-----------------------------------------------------')
        for ch in [channels[i] for i in id_descending]:
            print(f'{ch}: {power_all[ch]}')

    def plot_sv(self, fmax=None):
        """Plots and saves the singular values plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        for sv in range(self.n_sv):
            ax.plot(self.f, self.S_val[sv, sv, :], label=f'SV{sv+1}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Singular Values')
        # ax.set_title('SV Plot')
        ax.set_yscale('log')
        ax.legend()
        ax.grid()
        if fmax:
            ax.set_xlim([0, fmax])
        fig.tight_layout()
        figname = 'SVD'
        fig.savefig(os.path.join(
            self.savepath_figures, self.test_name, figname))
        print(
            f'SV Plot figure saved as {figname} in {os.path.join(self.savepath_figures, self.test_name)}')
        return fig, ax


class UFFGenerator:
    """
    Class used to read and generate .uff data from CSV and metadata JSON files.

    Attributes:
        test_name (str): Name of the test.
        project_name (str): Name of the project.
        savedata_path (str): Path where the data is stored and .uff file will be saved.
    """

    def __init__(self, test_name: str, project_name: str, input_data_path: str, savedata_path: str):
        self.test_name = test_name
        self.project_name = project_name
        self.input_data_path = input_data_path
        self.savedata_path = savedata_path
        self.output_ufffile_fullpath = os.path.join(
            self.savedata_path, f'{self.test_name}.uff')
        self.input_data_fullpath = os.path.join(
            self.input_data_path, f'{self.test_name}.csv')
        self.input_info_fullpath = os.path.join(
            self.input_data_path, f'{self.test_name}_information.json')
        self.df = None
        self.metadata = None

    def check_files_existence(self) -> None:
        """Check if both data and metadata files exist in the savedata_path."""
        required_files = [self.input_data_fullpath, self.input_info_fullpath]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(
                    f"{file} not found in {self.savedata_path}")

    def read_files(self) -> None:
        """Read the data and metadata files."""
        self.df = pd.read_csv(self.input_data_fullpath, delimiter=';')
        with open(self.input_info_fullpath, 'r') as f:
            self.metadata = json.load(f)

    def remove_existing_uff_file(self) -> None:
        """Remove the .uff file if it already exists."""
        if os.path.exists(self.output_ufffile_fullpath):
            os.remove(self.output_ufffile_fullpath)

    def generate_uff_data(self) -> None:
        """Generate and write .uff data from the CSV and metadata."""
        uff = pyuff.UFF(self.output_ufffile_fullpath)

        time = self.df['t'].values
        date = self.metadata['test_date']
        channels = self.df.drop(columns=['t']).values
        num_channels = channels.shape[1]

        dt = time[1] - time[0]  # Assuming uniform time intervals

        num_samples = channels.shape[0]
        response_mode = 'NONE'
        func_type = 1  # Time Response
        # reference_entity = 'NONE'
        abscissa_spacing = 1  # Even (no abscissa values stored)
        ord_data_type = 2  # Real, single precision
        if self.metadata['data_type'].lower() == 'acceleration':
            data_type = 12  # Acceleration
            test_name = f'{self.test_name}: Acceleration Measurements'
        elif self.metadata['data_type'].lower() == 'strain':
            data_type = 3  # Strain
            test_name = f'{self.test_name}: Strain Measurements'
        data_class_ordinate_label = self.metadata['data_type']
        data_class_ordinate_units_label = self.metadata['units']

        for i in range(num_channels):
            label = f'Channel {i+1}'
            rsp_dir = 3  # Default to 'z'
            channel_data = {
                'type': 58,
                'id1': label,
                'id2': test_name,
                'id3': date,
                'id4': self.project_name,
                'id5': response_mode,
                'func_type': func_type,
                'rsp_node': 0,
                'rsp_dir': rsp_dir,
                'ref_node': 0,
                'ref_dir': 0,
                'ord_data_type': ord_data_type,
                'abscissa_spacing': abscissa_spacing,
                'number_of_ord_values': num_samples,
                'abscissa_min': time[0],
                'abscissa_inc': dt,
                'z_axis_value': 0.0,
                'data': channels[:, i],
                'x': time,
                'ord_data_char': {
                    'data_type': data_type,
                    'length_units_exp': 0,
                    'force_units_exp': 0,
                    'temp_units_exp': 0,
                    'data_class_ordinate_label': data_class_ordinate_label,
                    'data_class_ordinate_units_label': data_class_ordinate_units_label,
                },
                'abscissa_data_char': {
                    'data_type': 17,  # Time
                    'length_units_exp': 0,
                    'force_units_exp': 0,
                    'temp_units_exp': 0,
                    'data_class_abscissa_label': 'Time',
                    'data_class_abscissa_units_label': 's',
                },
                'orddenom_spec_data_type': 0,
            }
            uff.write_sets([channel_data])
        print(
            f"UFF data generated and saved to {self.output_ufffile_fullpath}")
