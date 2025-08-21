import torch
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .models import default_peak_finding_model, default_thresholding_model
import streamlit as st

def interpolate_and_smooth(y, target_length=244):
    x = np.linspace(0, 1, len(y))
    new_x = np.linspace(0, 1, target_length)
    
    if len(y) == target_length:
        final = y
    elif len(y) > target_length:
        interpolated_values = np.interp(new_x, x, y).astype(float)
        final = pd.Series(interpolated_values)
    elif len(y) < target_length:
        cs = CubicSpline(x, y)
        final = cs(new_x)

    return pd.Series(final)

def peak_finding(wave, peak_finding_model):

    # Prepare waveform
    waveform = interpolate_and_smooth(wave)
    
    waveform_torch = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Get prediction from model
    pk_outputs = peak_finding_model(waveform_torch)
    prediction = int(round(pk_outputs.detach().numpy()[0][0], 0))

    # Apply Gaussian smoothing
    smoothed_waveform = gaussian_filter1d(wave, sigma=1.0)

    # Find peaks and troughs
    n = 18
    t = 14
    start_point = prediction - 6
    smoothed_peaks, _ = find_peaks(smoothed_waveform[start_point:], distance=n)
    smoothed_troughs, _ = find_peaks(-smoothed_waveform, distance=t)
    sorted_indices = np.argsort(smoothed_waveform[smoothed_peaks+start_point])
    highest_smoothed_peaks = np.sort(smoothed_peaks[sorted_indices[-5:]] + start_point)
    relevant_troughs = np.array([])
    for p in range(len(highest_smoothed_peaks)):
        c = 0
        for t in smoothed_troughs:
            if t > highest_smoothed_peaks[p]:
                if p != 4:
                    try:
                        if t < highest_smoothed_peaks[p+1]:
                            relevant_troughs = np.append(relevant_troughs, int(t))
                            break
                    except IndexError:
                        pass
                else:
                    relevant_troughs = np.append(relevant_troughs, int(t))
                    break
    relevant_troughs = relevant_troughs.astype('i')
    return highest_smoothed_peaks, relevant_troughs

def calculate_hearing_threshold(df, freq):
    file_name = getattr(df, 'name', 'unknown_file')
    cache_key = f"threshold_{file_name}_{freq}"

    if 'calculated_thresholds' not in st.session_state:
        st.session_state.calculated_thresholds = {}
    
    if 'manual_thresholds' in st.session_state:
        manual_cache_key = f"{file_name}_{freq}"
        if manual_cache_key in st.session_state.manual_thresholds:
            return st.session_state.manual_thresholds[manual_cache_key]
    
    if cache_key in st.session_state.calculated_thresholds:
        return st.session_state.calculated_thresholds[cache_key]
    
    atten = st.session_state.get('atten', False)
    db_column = 'Level(dB)' if not atten else 'PostAtten(dB)'

    thresholding_model = default_thresholding_model()
    
    # Filter DataFrame to include only data for the specified frequency
    df_filtered = df[df['Freq(Hz)'] == freq]

    # Get unique dB levels for the filtered DataFrame
    db_levels = sorted(df_filtered[db_column].unique(), reverse=True) if db_column == 'Level(dB)' else sorted(df_filtered[db_column].unique())
    waves = []

    for db in db_levels:
        khz = df_filtered[df_filtered[db_column] == db] #np.abs(db)]
        if not khz.empty:
            index = khz.index.values[-1]
            final = df_filtered.loc[index, '0':].dropna()
            final = pd.to_numeric(final, errors='coerce')
            final = np.array(final, dtype=np.float64)

            tenms = int((10/st.session_state.time_scale)*len(final))
            final = interpolate_and_smooth(final[:tenms], 244)
            # final = interpolate_and_smooth(final[:244])
            final *= st.session_state.multiply_y_factor

            if st.session_state.units == 'Nanovolts':
                final /= 1000

            waves.append(final)
    
    waves = np.array(waves)
    flattened_data = waves.flatten().reshape(-1, 1)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(flattened_data)

    # Step 2: Apply min-max scaling
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # Adjust range if needed
    scaled_data = min_max_scaler.fit_transform(standardized_data).reshape(waves.shape)
    waves = np.expand_dims(scaled_data, axis=2)
    
    # Perform prediction
    prediction = thresholding_model.predict(waves)
    y_pred = (prediction > 0.5).astype(int).flatten()

    if db_column == 'PostAtten(dB)':
        db_levels = np.array(db_levels)
        calibration_level = np.full(len(db_levels), st.session_state.calibration_levels[(df.name, freq)])
        db_levels = calibration_level - db_levels

    lowest_db = db_levels[0]
    previous_prediction = None


    for p, d in zip(y_pred, db_levels):
        if p == 0:
            if previous_prediction == 0:
                break
            previous_prediction = p
        else:
            lowest_db = d
            previous_prediction = p
    st.session_state.calculated_thresholds[cache_key] = lowest_db

    return lowest_db

def display_threshold_table(selected_dfs, selected_files, freqs):
    metrics_data = {'File Name': [], 'Frequency (Hz)': [], 'Estimated Threshold': []}
    
    for file_df, file_name in zip(selected_dfs, selected_files):
        for freq in freqs:
            if len(file_df[file_df['Freq(Hz)'] == freq]) == 0:
                continue
            try:
                threshold = calculate_hearing_threshold(file_df, freq)                
            except:
                threshold = np.nan
                pass
            #print(file_name)
            metrics_data['File Name'].append(file_name.split("/")[-1])
            metrics_data['Frequency (Hz)'].append(freq)
            metrics_data['Estimated Threshold'].append(threshold)
    
    metrics_table = pd.DataFrame(metrics_data)
    return metrics_table

def display_peaks_table(selected_dfs, selected_files, freqs, db_levels):
    metrics_data = {'File Name': [], 'Frequency (Hz)': [], 'Sound amplitude (dB SPL)': [],} 
    atten = st.session_state.get('atten', False)
    db_column = 'Level(dB)' if not atten else 'PostAtten(dB)'
    if atten:
        metrics_data = {**metrics_data, 'Attenuation (dB)':[], 'Calibration Level (dB)': []}

    ru = 'μV'
    if st.session_state.return_units == 'Nanovolts':
        ru = 'nV'

    metrics_data = {**metrics_data, f'Wave I amplitude (P1-T1) ({ru})': [], 'Latency to First Peak (ms)': [],
                    'Amplitude Ratio (Peak1/Peak4)': []}
    if st.session_state.all_peaks:
        metrics_data = {**metrics_data,
                        f'Peak 1 ({ru})': [], 'Peak 1 latency (ms)': [], f'Trough 1 ({ru})': [], 'Trough 1 latency (ms)': [],
                        f'Peak 2 ({ru})': [], 'Peak 2 latency (ms)': [], f'Trough 2 ({ru})': [], 'Trough 2 latency (ms)': [],
                        f'Peak 3 ({ru})': [], 'Peak 3 latency (ms)': [], f'Trough 3 ({ru})': [], 'Trough 3 latency (ms)': [],
                        f'Peak 4 ({ru})': [], 'Peak 4 latency (ms)': [], f'Trough 4 ({ru})': [], 'Trough 4 latency (ms)': [],
                        f'Peak 5 ({ru})': [], 'Peak 5 latency (ms)': [], f'Trough 5 ({ru})': [], 'Trough 5 latency (ms)': [],
                        }

    for file_df, file_name in zip(selected_dfs, selected_files):
        for freq in freqs:
            if len(file_df[file_df['Freq(Hz)'] == freq]) == 0:
                continue
            
            for db in db_levels:
                _, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db)
                if y_values is None:
                    continue
                if st.session_state.return_units == 'Nanovolts':
                    y_values *= 1000

                if highest_peaks is not None:
                    if highest_peaks.size > 0:  # Check if highest_peaks is not empty
                        first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                        latency_to_first_peak = highest_peaks[0] * (st.session_state.time_scale / len(y_values))

                        if len(highest_peaks) >= 4 and len(relevant_troughs) >= 4:
                            amplitude_ratio = (y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]) / (
                                        y_values[highest_peaks[3]] - y_values[relevant_troughs[3]])
                        else:
                            amplitude_ratio = np.nan

                        metrics_data['File Name'].append(file_name.split("/")[-1])
                        metrics_data['Frequency (Hz)'].append(freq)
                        if db_column == 'Level(dB)':
                            metrics_data['Sound amplitude (dB SPL)'].append(db)
                        else:
                            metrics_data['Sound amplitude (dB SPL)'].append(st.session_state.calibration_levels[(file_df.name, freq)] - db)
                            metrics_data['Attenuation (dB)'].append(db)
                            metrics_data['Calibration Level (dB)'].append(st.session_state.calibration_levels[(file_df.name, freq)])
                    
                        metrics_data[f'Wave I amplitude (P1-T1) ({ru})'].append(first_peak_amplitude)
                        metrics_data['Latency to First Peak (ms)'].append(latency_to_first_peak)
                        metrics_data['Amplitude Ratio (Peak1/Peak4)'].append(amplitude_ratio)

                        if st.session_state.all_peaks:
                            for pk_n in range(1, 6):  # Get up to 5 peaks for metrics
                                peak = highest_peaks[pk_n - 1] if pk_n <= len(highest_peaks) else np.nan
                                trough = relevant_troughs[pk_n - 1] if pk_n <= len(relevant_troughs) else np.nan
                                metrics_data[f'Peak {pk_n} ({ru})'].append(y_values[peak] if not np.isnan(peak) else np.nan)
                                metrics_data[f'Peak {pk_n} latency (ms)'].append(peak * (st.session_state.time_scale / len(y_values)))
                                metrics_data[f'Trough {pk_n} ({ru})'].append(y_values[trough] if not np.isnan(trough) else np.nan)
                                metrics_data[f'Trough {pk_n} latency (ms)'].append(trough * (st.session_state.time_scale / len(y_values)))

    metrics_table = pd.DataFrame(metrics_data)
    return metrics_table

def display_metrics_table_all_db(selected_dfs, selected_files, freqs, db_levels):
    output_pks = False if db_levels is None else True

    metrics_data = {'File Name': [], 'Frequency (Hz)': [], 'Sound amplitude (dB SPL)': [],} if output_pks else {'File Name': []}
    atten = st.session_state.get('atten', False)
    db_column = 'Level(dB)' if not atten else 'PostAtten(dB)'
    if atten:
        metrics_data = {**metrics_data, 'Attenuation (dB)':[], 'Calibration Level (dB)': []}

    ru = 'μV'
    if st.session_state.return_units == 'Nanovolts':
        ru = 'nV'

    metrics_data = {**metrics_data, 'Estimated Threshold': []}
    if output_pks:
        metrics_data = {**metrics_data, f'Wave I amplitude (P1-T1) ({ru})': [], 'Latency to First Peak (ms)': [],
                        'Amplitude Ratio (Peak1/Peak4)': []}
        if st.session_state.all_peaks:
            metrics_data = {**metrics_data,
                            f'Peak 1 ({ru})': [], 'Peak 1 latency (ms)': [], f'Trough 1 ({ru})': [], 'Trough 1 latency (ms)': [],
                            f'Peak 2 ({ru})': [], 'Peak 2 latency (ms)': [], f'Trough 2 ({ru})': [], 'Trough 2 latency (ms)': [],
                            f'Peak 3 ({ru})': [], 'Peak 3 latency (ms)': [], f'Trough 3 ({ru})': [], 'Trough 3 latency (ms)': [],
                            f'Peak 4 ({ru})': [], 'Peak 4 latency (ms)': [], f'Trough 4 ({ru})': [], 'Trough 4 latency (ms)': [],
                            f'Peak 5 ({ru})': [], 'Peak 5 latency (ms)': [], f'Trough 5 ({ru})': [], 'Trough 5 latency (ms)': [],
                            }

    for file_df, file_name in zip(selected_dfs, selected_files):
        for freq in freqs:
            if len(file_df[file_df['Freq(Hz)'] == freq]) == 0:
                continue
            try:
                threshold = calculate_hearing_threshold(file_df, freq)                
            except:
                threshold = np.nan
                pass
            
            if not output_pks:
                metrics_data['File Name'].append(file_name.split("/")[-1])
                metrics_data['Estimated Threshold'].append(threshold)
                continue
            for db in db_levels:
                _, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db)
                if y_values is None:
                    continue
                if st.session_state.return_units == 'Nanovolts':
                    y_values *= 1000

                if highest_peaks is not None:
                    if highest_peaks.size > 0:  # Check if highest_peaks is not empty
                        first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                        latency_to_first_peak = highest_peaks[0] * (st.session_state.time_scale / len(y_values))

                        if len(highest_peaks) >= 4 and len(relevant_troughs) >= 4:
                            amplitude_ratio = (y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]) / (
                                        y_values[highest_peaks[3]] - y_values[relevant_troughs[3]])
                        else:
                            amplitude_ratio = np.nan

                        metrics_data['File Name'].append(file_name.split("/")[-1])
                        metrics_data['Estimated Threshold'].append(threshold)
                        metrics_data['Frequency (Hz)'].append(freq)
                        if db_column == 'Level(dB)':
                            metrics_data['Sound amplitude (dB SPL)'].append(db)
                        else:
                            metrics_data['Sound amplitude (dB SPL)'].append(st.session_state.calibration_levels[(file_df.name, freq)] - db)
                            metrics_data['Attenuation (dB)'].append(db)
                            metrics_data['Calibration Level (dB)'].append(st.session_state.calibration_levels[(file_df.name, freq)])
                    
                        metrics_data[f'Wave I amplitude (P1-T1) ({ru})'].append(first_peak_amplitude)
                        metrics_data['Latency to First Peak (ms)'].append(latency_to_first_peak)
                        metrics_data['Amplitude Ratio (Peak1/Peak4)'].append(amplitude_ratio)

                        if st.session_state.all_peaks:
                            for pk_n in range(1, 6):  # Get up to 5 peaks for metrics
                                peak = highest_peaks[pk_n - 1] if pk_n <= len(highest_peaks) else np.nan
                                trough = relevant_troughs[pk_n - 1] if pk_n <= len(relevant_troughs) else np.nan
                                metrics_data[f'Peak {pk_n} ({ru})'].append(y_values[peak] if not np.isnan(peak) else np.nan)
                                metrics_data[f'Peak {pk_n} latency (ms)'].append(peak * (st.session_state.time_scale / len(y_values)))
                                metrics_data[f'Trough {pk_n} ({ru})'].append(y_values[trough] if not np.isnan(trough) else np.nan)
                                metrics_data[f'Trough {pk_n} latency (ms)'].append(trough * (st.session_state.time_scale / len(y_values)))

    metrics_table = pd.DataFrame(metrics_data)
    return metrics_table

def calculate_and_plot_wave(df, freq, db, peak_finding_model=default_peak_finding_model()):
    file_name = getattr(df, 'name', 'unknown_file')
    smooth_on = st.session_state.get('smooth_on', True)

    cache_key = f"wave_{file_name}_{freq}_{db}_{smooth_on}" 

    if 'calculated_waves' not in st.session_state:
        st.session_state.calculated_waves = {}

    if cache_key in st.session_state.calculated_waves:
        return st.session_state.calculated_waves[cache_key]

    if smooth_on:
        result = calculate_and_plot_wave_orig(df, freq, db, peak_finding_model)
    else:
        result = calculate_and_plot_wave_exact(df, freq, db, peak_finding_model)

    st.session_state.calculated_waves[cache_key] = result
    return result

def calculate_and_plot_wave_exact(df, freq, db, peak_finding_model=default_peak_finding_model(), 
                                  ):
    atten = st.session_state.get('atten', False)

    db_column = 'Level(dB)' if not atten else 'PostAtten(dB)'
    khz = df[(df['Freq(Hz)'] == freq) & (df[db_column] == db)]

    if not khz.empty:
        index = khz.index.values[-1] # in .arfs, sometimes multiple recordings if one is repeated, often the last one is the best
        
        orig_y = df.loc[index, '0':].dropna()
        orig_y = pd.to_numeric(orig_y, errors='coerce')#.dropna()
        orig_x = np.linspace(0, st.session_state.time_scale, len(orig_y))

        if st.session_state.units == 'Nanovolts':
            orig_y /= 1000

        orig_y *= st.session_state.multiply_y_factor

        # y_values for peak finding:
        tenms = int((10/st.session_state.time_scale)*len(orig_y)) if st.session_state.time_scale > 10 else len(orig_y)
        y_values_fpf = interpolate_and_smooth(orig_y[:tenms], 244)

        flattened_data = y_values_fpf.values.flatten().reshape(-1, 1)

        # Step 1: Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(flattened_data)

        # Step 2: Apply min-max scaling
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = min_max_scaler.fit_transform(standardized_data).reshape(y_values_fpf.shape)

        y_values_fpf = scaled_data

        highest_peaks, relevant_troughs = peak_finding(y_values_fpf, peak_finding_model)

        # convert back to original x-coordinates:
        if len(highest_peaks) > 0:
            highest_peaks = np.array([int((peak / 244) * tenms) for peak in highest_peaks])
            relevant_troughs = np.array([int((trough / 244) * tenms) for trough in relevant_troughs])
            # check for +/- interpolation errors:
            highest_peaks = np.array([np.argmax(orig_y[peak-1:peak+2]) + peak - 1 for peak in highest_peaks])
            relevant_troughs = np.array([np.argmin(orig_y[trough-1:trough+2]) + trough - 1 for trough in relevant_troughs])

        return orig_x, orig_y, highest_peaks, relevant_troughs
    else:
        return None, None, None, None
    
def calculate_and_plot_wave_orig(df, freq, db, peak_finding_model=default_peak_finding_model()):
    atten = st.session_state.get('atten', False)
    db_column = 'Level(dB)' if not atten else 'PostAtten(dB)'
    khz = df[(df['Freq(Hz)'] == freq) & (df[db_column] == db)]
    if not khz.empty:
        index = khz.index.values[-1]
        final = df.loc[index, '0':].dropna()
        final = pd.to_numeric(final, errors='coerce').dropna()

        target = int(244 * (st.session_state.time_scale / 10))

        y_values = interpolate_and_smooth(final, target)  # Original y-values for plotting
        sampling_rate = len(y_values) / st.session_state.time_scale

        x_values = np.linspace(0, len(y_values) / sampling_rate, len(y_values))

        #y_values = interpolate_and_smooth(final[:244])
        if st.session_state.units == 'Nanovolts':
            y_values /= 1000

        y_values *= st.session_state.multiply_y_factor

        y_values_fpf = interpolate_and_smooth(y_values[:244])

        # Flatten the data to scale all values across the group
        flattened_data = y_values_fpf.values.flatten().reshape(-1, 1)

        # Step 1: Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(flattened_data)

        # Step 2: Apply min-max scaling
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = min_max_scaler.fit_transform(standardized_data).reshape(y_values_fpf.shape)

        y_values_fpf = interpolate_and_smooth(scaled_data[:244])

        highest_peaks, relevant_troughs = peak_finding(y_values_fpf, peak_finding_model)

        return x_values, y_values, highest_peaks, relevant_troughs
    return None, None, None, None
