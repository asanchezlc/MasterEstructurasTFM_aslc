import copy
from dotenv import load_dotenv
import numpy as np
import os
import warnings

from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks, csd


def get_username():
    """
    Get the username from the .env file
    """
    load_dotenv()  # loads variables from .env file
    username = os.getenv('USERNAME')
    return username


def peaks_pos_neg(x, threshold=None, prominence=None, distance=None):
    peaks_a, _ = find_peaks(x, height=None, threshold=threshold, distance=distance, prominence=prominence,
                            width=None, wlen=None, rel_height=0.5, plateau_size=None)
    peaks_b, _ = find_peaks(-np.array(x), height=None, threshold=threshold, distance=distance,
                            prominence=prominence, width=None, wlen=None, rel_height=0.5,
                            plateau_size=None)
    peaks = np.hstack([peaks_a, peaks_b])

    return (peaks)


def interpolate_peaks(y_values, id_error):
    """
    Function Duties:
        Interpolate values in id_error with linear interpolation between
        previous and posterior correct values
    Input:
        y_values: values such that y_values[id_error] are incorrect
        id_error: index of wrong values
    Output:
        y_interpolated: y_values where y_interpolated[id_error] have been interpolated
    """
    y_interpolated = copy.deepcopy(y_values)
    for id_fil in id_error:
        # Get the previous and posterior id of correct values
        all_inf = np.arange(0, id_fil)
        only_val_post, only_val_prev = False, False
        if True in np.isin(all_inf, id_error, invert=True):
            id_prev = all_inf[np.isin(all_inf, id_error, invert=True)][-1]
        else:
            only_val_post, id_prev = True, -1
        all_sup = np.arange(id_fil+1, len(y_values)+1)
        if True in np.isin(all_sup, id_error, invert=True):
            id_post = all_sup[np.isin(all_sup, id_error, invert=True)][0]
        else:
            only_val_prev, id_post = True, len(y_values)
        if id_prev < 0:
            id_prev = 0
        if id_post >= len(y_values):
            id_post = len(y_values)-1

        # Get the previous and posterior values
        val_prev = y_values[id_prev]
        val_post = y_values[id_post]

        # Interpolate
        if id_fil == 0 or only_val_post:
            y_interpolated[id_fil] = val_post
        elif id_fil == len(y_values)-1 or only_val_prev:
            y_interpolated[id_fil] = val_prev
        else:
            y_interpolated[id_fil] = np.interp(
                id_fil, [id_prev, id_post], [val_prev, val_post])

    return y_interpolated


def interpolate_peaks_median(y_values, id_error, window=20):
    """
    Function Duties:
        Interpolate values in id_error and assign the median of the window
            (a kind of Hampfel filter defined by myself)
    Input:
        y_values: values such that y_values[id_error] are incorrect
        id_error: index of wrong values
    Output:
        y_interpolated: y_values where y_interpolated[id_error] have been interpolated
    """
    y_interpolated = copy.deepcopy(y_values)
    for id_fil in id_error:
        id_prev = id_fil-1
        id_post = id_fil+1

        # Encontramos anterior y posterior correctos (podría pasar que haya dos consecutivos erróneos)
        while (id_prev in id_error):
            id_prev += -1
        while (id_post in id_error):
            id_post += 1

        # Calculate the median of the window
        window_i = np.arange(id_prev-window//2, id_post+window//2)
        median_i = np.median(y_values[window_i[~np.isin(window_i, id_error)]])
        y_interpolated[id_fil] = median_i

    return y_interpolated

# COMMENTED LINES: APPLYING INVERTED HANNING WINDOW (NOT USED)
# peaks_all = np.sort(peaks_all)
# t_hann = 2  # s (full window's length)
# f = 80  # Hz
# length = int(t_hann*f)
# filter = np.zeros((len(peaks_all), len(y)))
# length_all = np.zeros(len(peaks_all))
# fig, ax = plt.subplots(1)
# for i in range(len(peaks_all)):
#     filter[i, :] = np.ones(len(y))
#     if i == 0:
#         length_i = min(length, 2*peaks_all[i], 2*(peaks_all[i+1]-peaks_all[i]))
#     elif i == len(peaks_all)-1:
#         length_i = min(length, 2*(len(y)-peaks_all[i]), 2*(peaks_all[i]-peaks_all[i-1]))
#     else:
#         length_i = min(length, 2*(peaks_all[i+1]-peaks_all[i]), 2*(peaks_all[i]-peaks_all[i-1]))
#     if np.mod(length_i, 2) == 0:  # hanning window must be odd so that 0 will be in the middle
#         length_i = length_i + 1
#     filter[i, peaks_all[i]-int(length_i/2):peaks_all[i]+int(length_i/2)+1] = 1-np.hanning(length_i)
#     length_all[i] = length_i
#     ax.plot(filter[i, :])
# filter_def = np.prod(filter, axis=0)

# inv_hann = 1-np.hanning(length)
# window = np.ones(len(y))


def manual_filter(y, distance):
    """
    Filter values further than "distance" to the average
    """
    peaks_all = np.array([], dtype=int)
    peaks = np.where(y > np.mean(y + distance))[0]
    peaks_all = np.hstack((peaks_all, peaks))
    peaks = np.where(y < np.mean(y - distance))[0]
    peaks_all = np.hstack((peaks_all, peaks))
    if len(peaks_all) > len(y)/4:
        raise ValueError('Too many peaks detected. Check the distance parameter and the quality of the signal.')
    y_interpolated = interpolate_peaks(y, peaks_all)
    # y_interpolated = interpolate_peaks_median(y, peaks_all)

    return y_interpolated, peaks_all


def read_config_file(file_path):
    """
    Read .txt containing metadata
    """
    config = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip whitespace and skip comments and empty lines
            line = line.strip()
            if '# Additional' in line:
                break
            if not line or line.startswith('#'):
                if 'Path' in line or 'path' in line:
                    is_path = True
                else:
                    is_path = False
                continue
            # Split the line into key and value
            key, value = line.split(';', 1)
            key = key.strip()
            value = value.strip()
            # Handle empty value case
            if value:
                if key == 'GF':
                    config[key] = float(value)
                elif key == 'ch_to_delete':
                    if len(value) > 0:
                        sg = value.split(',')
                        sg = [int(i) for i in sg]
                    config[key] = sg
                elif is_path:
                    config[key] = value
                elif 'pile' in key:
                    if 'damaged_piles' not in list(config):
                        config['damaged_piles'] = {}
                    string_list = value.split(',')
                    float_list = [float(item.strip()) for item in string_list]
                    float_list[0] = int(float_list[0])
                    config['damaged_piles'][key] = {'level': float_list[0],
                                                    'x': float_list[1],
                                                    'y': float_list[2],
                                                    'z': float_list[3]}
                else:
                    try:
                        config[key] = int(value)
                    except:
                        config[key] = float(value)
            else:
                config[key] = None
    return config


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
            f, Pxy = csd(data[ch_i], data[ch_j], fs=sps, nperseg=nxseg,
                         noverlap=noverlap, window=window)
            PSD_matr[i, j, :] = Pxy

    S_val = np.zeros((len(PSD_matr), len(PSD_matr), int((nxseg)/2+1)))
    for i in range(np.shape(PSD_matr)[2]):
        U1, S1, _V1_t = np.linalg.svd(PSD_matr[:, :, i])
        U1_1 = np.transpose(U1)
        S1 = np.diag(S1)
        S_val[:, :, i] = S1

    return (f, PSD_matr), (S_val, U1_1)


def butterworth_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut / nyquist, 1 - 0.01)
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)

    return y


def get_file_modification_date(file_path: str) -> str:
    """Get the last modification date of a file."""
    modification_time = os.path.getmtime(file_path)
    modification_date = datetime.fromtimestamp(
        modification_time).strftime('%Y-%m-%d %H:%M:%S')
    return modification_date
