
import numpy as np
import pandas as pd
import scipy

from scipy.signal import savgol_filter

"""
OUTILS FROM PROJECT DOCTORADO_CODES/ARDUINO

Functions defined there are useful so they are copy-pasted here
"""


def strain(F, E, A):
    eps = F/(E*A)
    return eps


def strain_cantilever(P, L, h, E, I):
    eps = P*L*h/(2*E*I)
    return eps


def read_decimals(measurement):
    """
    Function duties:
        It corrects an issue related to serial plot:
        when data is saved, decimals are not properly
        adjusted. Example: if we are saving 2 decimals,
        following timestamp will be obtained:
        ...1.7,1.8,1.9,1.10,1.11,1.12... 1.99.
        But we need:: ...1.07,1.08,1.09,1.10,1.11,1.12... 1.99.
    Input:
        measurement: list of strings.
            It contains data with lacking zeros
    Output:
        final_measurement: list of float.
            It contains data with zeros properly inserted
    """
    # Adjust time
    integers = []
    decimals = []
    for i in measurement:
        if not (isinstance(i, str)):
            if np.isnan(i):
                i = '0.0'  # assign 0 where there are no recording
        parts = i.split('.')
        integers.append(parts[0])
        decimals.append(parts[1])

    max_decimals = max([len(i) for i in decimals])

    final_measurement = []
    for i in measurement:
        if not (isinstance(i, str)):
            if np.isnan(i):
                i = '0.0'  # assign 0 where there are no recording
        parts = i.split('.')
        n_decimals = len(parts[1])
        zeros = (max_decimals-n_decimals)*'0'
        number = parts[0] + '.' + zeros + parts[1]
        final_measurement.append(float(number))

    return final_measurement


def DAC(measurement, n_bits=24, Vs=5, gain=128):
    """
    Function duties:
        Converts number from ADC into voltage
    Input:
        measurement: measurement made
        n_bits: 24 for HX711
        Vs: 3.3V or 5V
    """
    scale_neg = 0.5*Vs/gain  # voltage is between -20 and 20 mV
    resolution = Vs/(2**n_bits - 1)
    V_out = measurement*resolution
    V_out /= gain - scale_neg  # from [0, Vs/gain] to [-Vs/gain, Vs/gain]

    return V_out


def V0_2_strain(V0, Vs=5, GF=2.08):
    """
    Function duties:
        Converts voltage into strain
    Input:
        V0: measured voltage
        Vs: source voltage
        GF: gauge factor
    Return:
        Strain (Quarter bridge configuration)
    """
    eps = -4*V0/Vs / (GF*(1+2*V0/Vs))

    return eps


def from_bits_2_acc(bit, n_bits=10, sensitivity=0.3, Vs=5,
                    centered_V=1.65, already_detrended=False):
    """
    Function duties:
        Converts bits from the Arduino Analog pin into accelerations
    Input:
        bit: number read by Arduino
        n_bits: number of bits of the ADC
        sensitivity: V/g (from datasheet)
        Vs: Input voltage
        centered_V: output voltage for 0 accelerations (from datasheet)
        already_detrended=False; if True, centered_V is ignored
    Return:

    """
    resolution = Vs/(2**n_bits-1)
    V = bit * resolution
    if already_detrended:
        acc_g = V/sensitivity
    else:
        acc_g = (V-centered_V)/sensitivity
    acc = acc_g * 9.807

    return acc


def str_to_bool(input_str):
    if input_str.lower() == 'true':
        return True
    elif input_str.lower() == 'false':
        return False


def arrange_data(files, metadata_files, CHANNEL='Channel 1'):
    """
    Function Duties:
        Reads data from files contained in a list and arranges it into 2 dictionaries.
    Remark:
        The logic of this function is that multiple tests will be analyzed jointly,
        so all of them must be read and arranged in the same way.
    """
    results = dict()
    metadata = {'beam_data': {}, 'test_info': {},
                'steps_all': [], 'w_test_all': [],
                'unload_ref_all': [], 'higher_bits_when_load': [],
                'window': [], 'prominence': [], 'distance': []}

    for k, file in enumerate(files):
        meas = pd.read_csv(file, dtype=str)
        metadata_path = metadata_files[k]
        meta = read_metadata(metadata_path)

        # Arrange measurements in a dictionary
        results[str(k+1)] = dict()
        results[str(k+1)]['raw'] = dict()
        results[str(k+1)]['raw']['timestamp'] = read_decimals(list(meas['timestamp']))
        results[str(k+1)]['raw']['data'] = read_decimals(list(meas[CHANNEL]))
        results[str(k+1)]['timestamp'] = read_decimals(list(meas['timestamp']))
        # this will be without outliers
        results[str(k+1)]['measurement'] = read_decimals(list(meas[CHANNEL]))

        # Arrange metadata
        if str_to_bool(meta['isBeam']):
            type_test = 'cantilever'
            metadata['beam_data'][str(k+1)] = {'L': float(meta['L']), 'h': float(meta['h']),
                                               'b': float(meta['b']), 'E': float(meta['E'])}
        else:
            type_test = 'tension'
            metadata['beam_data'][str(k+1)] = {'E': float(meta['E']),
                                               'A': float(meta['A'])}
        metadata['test_info'][str(k+1)] = {'type_test': type_test,
                                           'tol': meta['tol'],
                                           'test_num': (file.split('_test')[1]).split('.txt')[0]}
        w_test = [9.81 * float(meta[item].split(' ')[0])
                  for item in meta if ('w' in item and len(item) < 4)]  # set to N from kg!
        if meta['step_type'] == 'Default':
            steps = [1 for _, _ in enumerate(w_test)]
        else:
            steps = [int(meta[item])
                     for item in meta if ('s' in item and len(item) < 3)]
        unload_ref = int(meta['unload_ref'])
        higher_bits_when_load = str_to_bool(meta['possitive_force'])

        metadata['w_test_all'].append(w_test)
        metadata['steps_all'].append(steps)
        metadata['unload_ref_all'].append(unload_ref)
        metadata['higher_bits_when_load'].append(higher_bits_when_load)
        metadata['window'].append(int(meta['window']))
        metadata['prominence'].append(int(meta['prominence']))
        metadata['distance'].append(int(meta['distance']))

    return metadata, results


def get_eps_ratios(results, metadata, eps_ref=10**-6):
    """
    Function Duties:
        Retrieve ratios of bits/force and bits/eps for each test
    """
    for k in list(results):
        type_test = metadata['test_info'][k]['type_test']
        beam_data = metadata['beam_data'][k]
        if type_test == 'cantilever':
            L, h, b, E = beam_data['L'], beam_data['h'], beam_data['b'], beam_data['E']
            I = 1/12*b*h**3
        else:
            E, A = beam_data['E'], beam_data['A']
        results[k]['steps']['ratio_f'] = dict()
        results[k]['steps']['ratio_eps_ref'] = dict()
        measurement = np.array(results[k]['filtered'])
        for w in list(results[k]['steps']['load']):
            ratio_f_all = []
            ratio_eps_ref_all = []
            for s in list(results[k]['steps']['load'][w]):
                weight = results[k]['steps']['applied_w'][w]
                deformed_ext = np.array(results[k]['steps']['load'][w][s])  # outer index
                deformed = measurement[np.arange(deformed_ext[0], deformed_ext[-1]+1)]  # value (bits)
                undeformed_ext = np.array(results[k]['steps']['zero'][w][s])  # index
                undeformed = measurement[np.arange(undeformed_ext[0], undeformed_ext[-1]+1)]  # value (bits)
                # Ratio: bits/force
                ratio_f = np.abs((np.mean(deformed) - np.mean(undeformed))/weight)
                # Ratio: bits/eps (for theoretical eps)
                if type_test == 'tension':
                    eps = strain(weight, E, A)  # theor. value for tension
                elif type_test == 'cantilever':
                    eps = strain_cantilever(weight, L, h, E, I)  # theor. value for cantilever
                ratio_eps = np.abs((np.mean(deformed) - np.mean(undeformed))/eps)
                ratio_eps_ref = ratio_eps * eps_ref  # bits per eps for scaled
                ratio_f_all.append(ratio_f)
                ratio_eps_ref_all.append(ratio_eps_ref)
            results[k]['steps']['ratio_f'][w] = ratio_f_all
            results[k]['steps']['ratio_eps_ref'][w] = ratio_eps_ref_all
    return results

# fig, ax = plt.subplots(1)
# timestamp = np.array(results[k]['timestamp'])
# ax.plot(timestamp, measurement)
# ax.scatter(timestamp[deformed_ext], measurement[deformed_ext])
# ax.scatter(timestamp[undeformed_ext], measurement[undeformed_ext])


def get_ratios_all_tests(results):
    ratios = dict()
    for k in list(results):
        ratios[k] = list()
        for w in list(results[k]['steps']['load']):
            for s in list(results[k]['steps']['load'][w]):
                ratios[k].append(results[k]['steps']['ratio_eps_ref'][s])
    return ratios


def get_steps_delimitations(results, metadata):
    """
    Function Duties:
    This function is quite long because at the first time, multiple
    steps were associated to a single force. It is now maintained, although
    it is much easier to have defined a single step for each force, so this
    function could be much more simpler
    """

    for j, k in enumerate(list(results)):
        # General data
        unload_ref = metadata['unload_ref_all'][j]
        steps = metadata['steps_all'][j]
        higher_bits_when_load_all = metadata['higher_bits_when_load'][j]
        w_test = metadata['w_test_all'][j]

        measurement = np.array(results[k]['filtered'])
        peaks = np.array(results[k]['peaks_id'])

        # Get peaks above and below unload_ref
        if higher_bits_when_load_all:
            load_peaks = peaks[np.where(measurement[peaks] > unload_ref)]
            unload_peaks = peaks[np.where(measurement[peaks] <= unload_ref)]
        else:
            unload_peaks = peaks[np.where(measurement[peaks] > unload_ref)]
            load_peaks = peaks[np.where(measurement[peaks] <= unload_ref)]
        results[k]['steps'] = dict()
        results[k]['steps']['zero'] = dict()

        # Get delimitations for steps of zero load
        s, w = 0, 0  # step and weight
        for i, _ in enumerate(unload_peaks):
            if i//2 >= np.sum(steps[0:w+1]):
                w += 1
                s = 0
            if np.mod(i, 2) == 0:
                continue
            elif i == len(unload_peaks) - 1:
                break
            if 'w' + str(w+1) not in list(results[k]['steps']['zero']):
                results[k]['steps']['zero']['w' + str(w+1)] = dict()
            results[k]['steps']['zero']['w' +
                                        str(w+1)]['s' + str(s+1)] = [unload_peaks[i], unload_peaks[i+1]]
            s += 1

        # Get delimitations for loaded
        results[k]['steps']['load'] = dict()
        s, w = 0, 0  # step and weight
        for i, _ in enumerate(load_peaks):
            if i//2 >= np.sum(steps[0:w+1]):
                w += 1
                s = 0
            if np.mod(i, 2) != 0:
                continue
            elif i == len(load_peaks) - 1:
                break
            if 'w' + str(w+1) not in list(results[k]['steps']['load']):
                results[k]['steps']['load']['w' + str(w+1)] = dict()
            results[k]['steps']['load']['w' +
                                        str(w+1)]['s' + str(s+1)] = [load_peaks[i], load_peaks[i+1]]
            s += 1

        # Add applied forces
        results[k]['steps']['applied_w'] = dict()
        for ii, w in enumerate(w_test):
            results[k]['steps']['applied_w']['w'+str(ii+1)] = w

    return results


def remove_outliers_set_t0(results, apply_outliers_filter=True,
                           n_loop_filter=3, threshold=8000, prominence=8000):
    """
    Function Duties:
        - Remove outliers from the data
        - Set the initial time to zero
    Input:
        apply_outliers_filter: if True, outliers are removed; if False, no outliers
            filter is applied
        n_loop_filter: higher number -> more outliers are removed
        threshold: threshold for peaks
        prominence: prominence for peaks
    Return:
        results: dictionary with the filtered data, having arranged
            the raw data independently
    """
    if not apply_outliers_filter:
        n_loop_filter = 1
    for i in range(n_loop_filter):
        for k in list(results):
            # Raw data
            if i == 1:
                t = results[k]['raw']['timestamp']
                results[str(k)]['raw']['timestamp'] = [j - t[0] for j in t]
            # Filtered data
            t = results[k]['timestamp']
            m = results[k]['measurement']
            if apply_outliers_filter:
                peaks = peaks_pos_neg(
                    m, threshold=threshold, prominence=prominence)
            else:
                peaks = []
            results[str(k)]['timestamp'] = [j - t[0]
                                            for i, j in enumerate(t) if i not in peaks]  # scale timestamp
            results[str(k)]['measurement'] = [
                j for i, j in enumerate(m) if i not in peaks]

    return results


def filter_signal_oscillation(results, window_length=500, polyorder=0):
    for k in list(results):
        filtered = savgol_filter(
            results[k]['measurement'], window_length, polyorder)
        results[str(k)]['filtered'] = filtered

    return results


def get_bits_ref(eps_ref, Vs=5, GF=2.08, n_bits=24, gain=128):
    """
    Function Duties:
        Get the number of bits required to produce a strain of value eps_ref
    """
    test_bits = np.arange(0, 10**5)
    eps_bits = [abs(V0_2_strain(DAC(i, n_bits=n_bits, Vs=Vs, gain=gain), Vs=Vs, GF=GF)) for i in test_bits]
    delta_eps = [np.abs(i-eps_ref) for i in eps_bits]
    bits_ref = test_bits[np.argmin(delta_eps)]
    return bits_ref


def peaks_pos_neg(x, threshold=None, prominence=None, distance=None):
    peaks_a, _ = scipy.signal.find_peaks(x, height=None, threshold=threshold, distance=distance, prominence=prominence,
                                         width=None, wlen=None, rel_height=0.5, plateau_size=None)
    peaks_b, _ = scipy.signal.find_peaks(-np.array(x), height=None, threshold=threshold, distance=distance,
                                         prominence=prominence, width=None, wlen=None, rel_height=0.5,
                                         plateau_size=None)
    peaks = np.hstack([peaks_a, peaks_b])

    return (peaks)


def detect_plateaus(results, metadata):
    """
    Function Duties:
        - Detect plateaus in the signal
        - Get the start and end of each plateau
    Input:
        results: dictionary with the filtered data
        window: window for detecting peaks
        prominence: prominence for detecting peaks
        distance: minimum distance between peaks
    Return:
        results: dictionary with the peaks detected
    """
    for j, k in enumerate(results):
        m = results[k]['filtered']
        window, prominence, distance = metadata['window'][j], metadata['prominence'][j], metadata['distance'][j]

        var_l, var_r = [], []  # variation of values from x to a point spaced window
        for i, x in enumerate(m):
            if i - window < 0:
                var_l.append(0)
                var_r.append(x - m[i + window])
            elif i + window > len(m) - 1:
                var_l.append(x - m[i - window])
                var_r.append(0)
            else:
                var_l.append(x - m[i - window])
                var_r.append(x - m[i + window])

        peaks_l = peaks_pos_neg(
            var_l, prominence=prominence, distance=distance)
        peaks_r = peaks_pos_neg(
            var_r, prominence=prominence, distance=distance)
        peaks = np.hstack([peaks_l, peaks_r])
        peaks = np.append(peaks, len(m) - 1)  # we add the final point
        peaks = np.sort(peaks)

        # Add first point if it is not in the zero-load part
        if metadata['higher_bits_when_load'][j]:
            if m[peaks[0]] > metadata['unload_ref_all'][j]:  # First peak point must be in the zero-load part
                peaks = np.insert(peaks, 0, 0)
        else:
            if m[peaks[0]] < metadata['unload_ref_all'][j]:  # First peak point must be in the zero-load part)
                peaks = np.insert(peaks, 0, 0)

        results[k]['peaks_id'] = peaks

    return results


def read_metadata(file_path):
    metadata = dict()
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace from the line
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            # Split the line into key and value parts on ';'
            parts = line.split(';')
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                # Check for numeric values that are masked as comments
                if '#' in value:  # Check if there is a comment in the value
                    # Remove the comment part
                    value = value.split('#')[0].strip()
                metadata[key] = value
    return metadata
