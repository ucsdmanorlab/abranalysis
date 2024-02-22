import streamlit as st
import fdasrsf as fs
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
import tempfile
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
import struct
from datetime import datetime
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def plot_scatter_waves(df, freq, db, background_curves=False, smoothing_method='None', sigma=3, n=15):
    fig = go.Figure()
    khz = df[(df['Freq(Hz)'].astype(float) == freq) & (df['Level(dB)'].astype(float) == db)]
    
    if not khz.empty:
        index = khz.index.values[0]
        final = df.loc[index, '0':]
        final = pd.to_numeric(final, errors='coerce')

        # Find highest peaks separated by at least n data points
        peaks, _ = find_peaks(final, distance=n)
        highest_peaks = peaks[np.argsort(final[peaks])[-5:]]

        if multiply_y_factor != 1:
            y_values = final * multiply_y_factor
        else:
            y_values = final

        # Plot scatter plot instead of line plot
        fig.add_trace(go.Scatter(x=np.arange(len(final)), y=y_values, mode='markers', name='Scatter Plot'))

        # Mark the highest peaks with red markers
        fig.add_trace(go.Scatter(x=highest_peaks, y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks'))

        # Annotate the peaks with red color, smaller font, and closer to the peaks
        for peak in highest_peaks:
            fig.add_annotation(
                x=peak,
                y=y_values[peak],
                text=f'{y_values[peak]:.4f}',
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                arrowwidth=2,
                ax=0,
                ay=-30,
                font=dict(color='red', size=10)
            )

    return fig

def plotting_waves_cubic_spline(df, freq=16000, db=90, n=45):
    khz = df[df['Freq(Hz)'] == freq]
    dbkhz = khz[khz['Level(dB)'] == db]
    index = dbkhz.index.values[0]
    original_waveform = df.loc[index, '0':]
    original_waveform = pd.to_numeric(original_waveform, errors='coerce')[:-1]

    if multiply_y_factor != 1:
        original_waveform *= multiply_y_factor

    # Apply cubic spline interpolation
    smooth_time = np.linspace(0, len(original_waveform) - 1, len(original_waveform)*3)
    cs = CubicSpline(np.arange(len(original_waveform)), original_waveform)
    smooth_amplitude = cs(smooth_time)

    # Find highest peaks separated by at least n data points in the smoothed curve
    peaks, _ = find_peaks(smooth_amplitude, distance=n)
    highest_peaks = peaks[np.argsort(smooth_amplitude[peaks])[-5:]]

    if highest_peaks.size > 0:  # Check if highest_peaks is not empty
        first_peak_value = smooth_amplitude[np.sort(highest_peaks)[0]]

        original_peaks, _ = find_peaks(original_waveform, distance=15)
        print(original_peaks)
        original_highest_peaks = original_peaks[np.argsort(original_waveform[original_peaks])[-5:]]
        if original_peaks.size > 0:
            first_original_peak_value = original_waveform[np.sort(original_highest_peaks)[0]]
            error = abs(first_original_peak_value - first_peak_value)
            print(original_highest_peaks)
            print(first_original_peak_value)
            print(first_peak_value)
            print(f"Error between the first peaks: {error}")

    # Create a plotly figure
    fig = go.Figure()

    # Plot the original ABR waveform
    fig.add_trace(go.Scatter(x=np.arange(len(original_waveform)), y=original_waveform, mode='lines', name='Original ABR', opacity=0.8))

    # Plot the cubic spline interpolation
    fig.add_trace(go.Scatter(x=smooth_time, y=smooth_amplitude, mode='lines', name='Cubic Spline Interpolation'))

    if highest_peaks.size > 0:
        first_peak = np.sort(highest_peaks)[0]
        fig.add_shape(
            type='line',
            x0=smooth_time[first_peak],
            x1=smooth_time[first_peak],
            y0=min(smooth_amplitude),
            y1=max(smooth_amplitude),
            line=dict(color='gray', dash='dash')
        )
        fig.add_trace(go.Scatter(x=[smooth_time[first_peak]], y=[np.nan], mode='markers', marker=dict(color='blue', opacity=0.5), name='First Peak'))

    # Set layout options
    fig.update_layout(title=f'{uploaded_file.name}', xaxis_title='Index', yaxis_title='Voltage (mV)', legend=dict(x=0, y=1, traceorder='normal'))

    # Show the plot using Streamlit
    return fig

def update_title_and_legend_if_single_frequency(fig, selected_freqs):
    if len(set(selected_freqs)) == 1:
        fig.update_layout(title=f'{uploaded_file.name} - Freq: {selected_freqs[0]} Hz')
        for trace in fig.data:
            if 'Freq' in trace.name:
                trace.name = trace.name.replace(f'Freq: {trace.name.split(" ")[1]} Hz, ', '')
    return fig

def plot_waves_single_frequency(df, freq, y_min, y_max, plot_time_warped=False):
    fig = go.Figure()
    
    # Get unique dB levels
    db_values = []

    waves_array = []  # Array to store all waves

    for db in range(0,95,5):
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.loc[index, '0':]
            final = pd.to_numeric(final, errors='coerce')

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final

            db_values.append(db)
            waves_array.append(y_values.to_list())  # Append the current wave to the array
    
    #waves_array = np.array(waves_array)
    waves_array = np.array([wave[:-1] for wave in waves_array])
    
    # Optionally apply time warping to all waves in the array
    if plot_time_warped:
        time = np.linspace(0, 10, waves_array.shape[1])
        obj = fs.fdawarp(np.array(waves_array).T, time)
        obj.srsf_align(parallel=True)
        waves_array = obj.fn.T  # Use the time-warped curves

    wave_colors = [f'rgb(255, {b}, {b})' for b in np.linspace(255, 100, len(db_values))]

    # Plot all waves in the array
    for i, (db, waves) in enumerate(zip(db_values, waves_array)):
        fig.add_trace(go.Scatter(x=np.linspace(0,10, waves_array.shape[1]), y=waves, mode='lines', name=f'dB: {db}', line=dict(color=wave_colors[i])))

    fig.update_layout(title=f'{uploaded_files[0].name} - Frequency: {freq} Hz, Predicted Threshold: {calculate_hearing_threshold(df, freq)} dB', xaxis_title='Time (ms)', yaxis_title='Voltage (mV)')
    fig.update_layout(annotations=annotations)
    fig.update_layout(yaxis_range=[y_min, y_max])

    return fig

def plot_waves_single_db(df, db, y_min, y_max):
    fig = go.Figure()

    for freq in sorted(df['Freq(Hz)'].unique()):
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.loc[index, '0':]
            final = pd.to_numeric(final, errors='coerce')

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final

            fig.add_trace(go.Scatter(x=np.linspace(0,10, len(y_values)), y=y_values, mode='lines', name=f'Frequency: {freq} Hz'))
            

    fig.update_layout(title=f'{uploaded_files[0].name} - dB Level: {db}', xaxis_title='Index', yaxis_title='Voltage (mV)')
    fig.update_layout(annotations=annotations)
    fig.update_layout(yaxis_range=[y_min, y_max])

    return fig

def plot_waves_single_tuple(df, freq, db, y_min, y_max):
    fig = go.Figure()
    i=0
    for df in dfs:
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.loc[index, '0':]
            final = pd.to_numeric(final, errors='coerce')

            time_axis = np.linspace(0, 10, len(final))

            # Find highest peaks separated by at least n data points

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final

            peaks, _ = find_peaks(y_values, distance=15)
            troughs, _ = find_peaks(-y_values, distance=15)
            highest_peaks = peaks[np.argsort(final[peaks])[-5:]]
            highest_peaks = np.sort(highest_peaks)
            relevant_troughs = np.array([])
            for p in range(len(highest_peaks)):
                c = 0
                for t in troughs:
                    if t > highest_peaks[p]:
                        if p != 4:
                            if t < highest_peaks[p+1]:
                                relevant_troughs = np.append(relevant_troughs, int(t))
                                break
                        else:
                            relevant_troughs = np.append(relevant_troughs, int(t))
                            break
            relevant_troughs = relevant_troughs.astype('i')

            fig.add_trace(go.Scatter(x=np.linspace(0,10, len(y_values)), y=y_values, mode='lines', name=f'{uploaded_files[i].name}'))
            i+=1

            # Mark the highest peaks with red markers
            peaks_trace = go.Scatter(x=time_axis[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks')
            fig.add_trace(peaks_trace)

            fig.add_trace(go.Scatter(x=time_axis[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs'))

    fig.update_layout(title=f'Freq = {freq}, dB = {db}', xaxis_title='Time (ms)', yaxis_title='Voltage (mV)')
    fig.update_layout(annotations=annotations)
    fig.update_layout(yaxis_range=[y_min, y_max])

    return fig

def plot_3d_surface(df, freq, y_min, y_max):
    fig = go.Figure()

    db_levels = sorted(df['Level(dB)'].unique())
    original_waves = []  # List to store original waves

    #wave_colors = ['rgb(255, 0, 0)', 'rgb(255, 128, 128)', 'rgb(255, 191, 191)', 'rgb(255, 224, 224)', 'rgb(255, 240, 240)']
    wave_colors = [f'rgb(255, 0, 255)' for b in np.linspace(0, 0, len(db_levels))]
    connecting_line_color = 'rgba(0, 255, 0, 0.3)'

    for db in db_levels:
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.loc[index, '0':]
            final = pd.to_numeric(final, errors='coerce')

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final

            original_waves.append(y_values.to_list())  # Append the current wave to the list

    # Convert original waves to a 2D numpy array
    original_waves_array = np.array([wave[:-1] for wave in original_waves])

    # Apply time warping to all waves in the array
    time = np.linspace(0, 10, original_waves_array.shape[1])
    obj = fs.fdawarp(original_waves_array.T, time)
    obj.srsf_align(parallel=True)
    warped_waves_array = obj.fn.T  # Use the time-warped curves

    # Plot all time-warped waves in the array
    for i, (db, warped_waves) in enumerate(zip(db_levels, warped_waves_array)):
        fig.add_trace(go.Scatter3d(x=[db] * len(warped_waves), y=np.linspace(0, 10, len(warped_waves)), z=warped_waves, mode='lines', name=f'dB: {db}', line=dict(color=wave_colors[i])))

    # Create surface connecting the curves at each time point
    for i in range(len(time)):
        z_values_at_time = [warped_waves_array[j, i] for j in range(len(db_levels))]
        fig.add_trace(go.Scatter3d(x=db_levels, y=[time[i]] * len(db_levels), z=z_values_at_time, mode='lines', name=f'Time: {time[i]:.2f} ms', line=dict(color=connecting_line_color)))

    fig.update_layout(title=f'{uploaded_files[0].name} - Frequency: {freq} Hz', scene=dict(xaxis_title='dB Level', yaxis_title='Time (ms)', zaxis_title='Voltage (mV)'))
    fig.update_layout(annotations=annotations)
    fig.update_layout(scene=dict(zaxis=dict(range=[y_min, y_max])))

    return fig

def display_metrics_table(df, freq, db, baseline_level):
    khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
    if not khz.empty:
        index = khz.index.values[0]
        final = df.loc[index, '0':]
        final = pd.to_numeric(final, errors='coerce')

        if multiply_y_factor != 1:
            y_values = final * multiply_y_factor
        else:
            y_values = final
        
        # Adjust the waveform by subtracting the baseline level
        y_values -= baseline_level

        # Find highest peaks separated by at least n data points
        peaks, _ = find_peaks(y_values, distance=15)
        troughs, _ = find_peaks(-y_values, distance=15)
        highest_peaks = peaks[np.argsort(final[peaks])[-5:]]
        highest_peaks = np.sort(highest_peaks)
        relevant_troughs = np.array([])
        for p in range(len(highest_peaks)):
            c = 0
            for t in troughs:
                if t > highest_peaks[p]:
                    if p != 4:
                        if t < highest_peaks[p+1]:
                            relevant_troughs = np.append(relevant_troughs, int(t))
                            break
                    else:
                        relevant_troughs = np.append(relevant_troughs, int(t))
                        break
        relevant_troughs = relevant_troughs.astype('i')

        if highest_peaks.size > 0:  # Check if highest_peaks is not empty
            first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
            latency_to_first_peak = highest_peaks[0] * (10 / len(y_values))  # Assuming 10 ms duration for waveform

            if len(highest_peaks) >= 4:
                amplitude_ratio = (y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]) / (y_values[highest_peaks[3]] - y_values[relevant_troughs[3]])
            else:
                amplitude_ratio = np.nan

            metrics_table = pd.DataFrame({
                'Metric': ['First Peak Amplitude (mV)', 'Latency to First Peak (ms)', 'Amplitude Ratio (Peak1/Peak4)', 'Estimated Threshold'],
                'Value': [first_peak_amplitude, latency_to_first_peak, amplitude_ratio, calculate_hearing_threshold(df, freq)],
            })
            st.table(metrics_table)

def display_metrics_table_all_db(df, freq, db_levels, baseline_level):
    metrics_data = {'dB Level': [], 'First Peak Amplitude (mV)': [], 'Latency to First Peak (ms)': [], 'Amplitude Ratio (Peak1/Peak4)': []}
    
    for db in db_levels:
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        if not khz.empty:
            index = khz.index.values[0]
            final = df.loc[index, '0':]
            final = pd.to_numeric(final, errors='coerce')

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final
            
            # Adjust the waveform by subtracting the baseline level
            y_values -= baseline_level

            # Find highest peaks separated by at least n data points
            peaks, _ = find_peaks(y_values, distance=15)
            troughs, _ = find_peaks(-y_values, distance=15)
            highest_peaks = peaks[np.argsort(final[peaks])[-5:]]
            highest_peaks = np.sort(highest_peaks)
            relevant_troughs = np.array([])
            for p in range(len(highest_peaks)):
                c = 0
                for t in troughs:
                    if t > highest_peaks[p]:
                        if p != 4:
                            if t < highest_peaks[p+1]:
                                relevant_troughs = np.append(relevant_troughs, int(t))
                                break
                        else:
                            relevant_troughs = np.append(relevant_troughs, int(t))
                            break
            relevant_troughs = relevant_troughs.astype('i')

            if highest_peaks.size > 0:  # Check if highest_peaks is not empty
                first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                latency_to_first_peak = highest_peaks[0] * (10 / len(y_values))  # Assuming 10 ms duration for waveform

                if len(highest_peaks) >= 4:
                    amplitude_ratio = (y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]) / (y_values[highest_peaks[3]] - y_values[relevant_troughs[3]])
                else:
                    amplitude_ratio = np.nan

                metrics_data['dB Level'].append(db)
                metrics_data['First Peak Amplitude (mV)'].append(first_peak_amplitude)
                metrics_data['Latency to First Peak (ms)'].append(latency_to_first_peak)
                metrics_data['Amplitude Ratio (Peak1/Peak4)'].append(amplitude_ratio)

    metrics_table = pd.DataFrame(metrics_data)
    st.table(metrics_table)

def plot_waves_stacked(df, freq, y_min, y_max, plot_time_warped=False):
    fig = go.Figure()

    # Get unique dB levels
    unique_dbs = sorted(df['Level(dB)'].unique())

    # Calculate the vertical offset for each waveform
    num_dbs = len(unique_dbs)
    vertical_spacing = (y_max - y_min) / num_dbs

    # Initialize an offset for each dB level
    db_offsets = {db: y_min + i * vertical_spacing for i, db in enumerate(unique_dbs)}

    # Process and plot each waveform
    for db in sorted(df['Level(dB)'].unique()):
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]

        if not khz.empty:
            index = khz.index.values[0]
            final = df.loc[index, '0':]
            final = pd.to_numeric(final, errors='coerce')

            # Apply the vertical offset
            y_values = final + db_offsets[db]

            # Optionally apply time warping
            if plot_time_warped:
                # ... (your time warping code here)
                pass

            # Plot the waveform
            fig.add_trace(go.Scatter(x=np.linspace(0, 10, y_values.shape[0]),
                                     y=y_values,
                                     mode='lines',
                                     name=f'dB: {db}',
                                     line=dict(color='black')))

    fig.update_layout(title=f'{uploaded_files[0].name} - Frequency: {freq} Hz',
                      xaxis_title='Time (ms)',
                      yaxis_title='Voltage (mV)')
    fig.update_layout(yaxis_range=[y_min, y_max])
    # Set custom width and height (in pixels)
    custom_width = 400
    custom_height = 700

    fig.update_layout(width=custom_width, height=custom_height)

    fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False))

    return fig

def arfread(PATH, **kwargs):
    # defaults
    PLOT = kwargs.get('PLOT', False)
    RP = kwargs.get('RP', False)
    
    isRZ = not RP
    
    data = {'RecHead': {}, 'groups': []}

    # open file
    with open(PATH, 'rb') as fid:
        # open RecHead data
        data['RecHead']['ftype'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['ngrps'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['nrecs'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['grpseek'] = struct.unpack('200i', fid.read(4*200))
        data['RecHead']['recseek'] = struct.unpack('2000i', fid.read(4*2000))
        data['RecHead']['file_ptr'] = struct.unpack('i', fid.read(4))[0]

        data['groups'] = []
        bFirstPass = True
        for x in range(data['RecHead']['ngrps']):
            # jump to the group location in the file
            fid.seek(data['RecHead']['grpseek'][x], 0)

            # open the group
            data['groups'].append({
                'grpn': struct.unpack('h', fid.read(2))[0],
                'frecn': struct.unpack('h', fid.read(2))[0],
                'nrecs': struct.unpack('h', fid.read(2))[0],
                'ID': get_str(fid.read(16)),
                'ref1': get_str(fid.read(16)),
                'ref2': get_str(fid.read(16)),
                'memo': get_str(fid.read(50)),
            })

            # read temporary timestamp
            if bFirstPass:
                if isRZ:
                    ttt = struct.unpack('q', fid.read(8))[0]
                    fid.seek(-8, 1)
                    data['fileType'] = 'BioSigRZ'
                else:
                    ttt = struct.unpack('I', fid.read(4))[0]
                    fid.seek(-4, 1)
                    data['fileType'] = 'BioSigRP'
                data['fileTime'] = datetime.utcfromtimestamp(ttt/86400 + datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')
                bFirstPass = False

            if isRZ:
                data['groups'][x]['beg_t'] = struct.unpack('q', fid.read(8))[0]
                data['groups'][x]['end_t'] = struct.unpack('q', fid.read(8))[0]
            else:
                data['groups'][x]['beg_t'] = struct.unpack('I', fid.read(4))[0]
                data['groups'][x]['end_t'] = struct.unpack('I', fid.read(4))[0]
            
            data['groups'][x].update({
                'sgfname1': get_str(fid.read(100)),
                'sgfname2': get_str(fid.read(100)),
                'VarName1': get_str(fid.read(15)),
                'VarName2': get_str(fid.read(15)),
                'VarName3': get_str(fid.read(15)),
                'VarName4': get_str(fid.read(15)),
                'VarName5': get_str(fid.read(15)),
                'VarName6': get_str(fid.read(15)),
                'VarName7': get_str(fid.read(15)),
                'VarName8': get_str(fid.read(15)),
                'VarName9': get_str(fid.read(15)),
                'VarName10': get_str(fid.read(15)),
                'VarUnit1': get_str(fid.read(5)),
                'VarUnit2': get_str(fid.read(5)),
                'VarUnit3': get_str(fid.read(5)),
                'VarUnit4': get_str(fid.read(5)),
                'VarUnit5': get_str(fid.read(5)),
                'VarUnit6': get_str(fid.read(5)),
                'VarUnit7': get_str(fid.read(5)),
                'VarUnit8': get_str(fid.read(5)),
                'VarUnit9': get_str(fid.read(5)),
                'VarUnit10': get_str(fid.read(5)),
                'SampPer_us': struct.unpack('f', fid.read(4))[0],
                'cc_t': struct.unpack('i', fid.read(4))[0],
                'version': struct.unpack('h', fid.read(2))[0],
                'postproc': struct.unpack('i', fid.read(4))[0],
                'dump': get_str(fid.read(92)),
                'recs': [],
            })

            for i in range(data['groups'][x]['nrecs']):
                record_data = {
                        'recn': struct.unpack('h', fid.read(2))[0],
                        'grpid': struct.unpack('h', fid.read(2))[0],
                        'grp_t': struct.unpack('q' if isRZ else 'I', fid.read(8))[0],
                        #'grp_d': datetime.utcfromtimestamp(data['groups'][x]['recs'][i]['grp_t']/86400 + datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S'),
                        'newgrp': struct.unpack('h', fid.read(2))[0],
                        'sgi': struct.unpack('h', fid.read(2))[0],
                        'chan': struct.unpack('B', fid.read(1))[0],
                        'rtype': get_str(fid.read(1)),
                        'npts': struct.unpack('H' if isRZ else 'h', fid.read(2))[0],
                        'osdel': struct.unpack('f', fid.read(4))[0],
                        'dur_ms': struct.unpack('f', fid.read(4))[0],
                        'SampPer_us': struct.unpack('f', fid.read(4))[0],
                        'artthresh': struct.unpack('f', fid.read(4))[0],
                        'gain': struct.unpack('f', fid.read(4))[0],
                        'accouple': struct.unpack('h', fid.read(2))[0],
                        'navgs': struct.unpack('h', fid.read(2))[0],
                        'narts': struct.unpack('h', fid.read(2))[0],
                        'beg_t': struct.unpack('q' if isRZ else 'I', fid.read(8))[0],
                        'end_t': struct.unpack('q' if isRZ else 'I', fid.read(8))[0],
                        'Var1': struct.unpack('f', fid.read(4))[0],
                        'Var2': struct.unpack('f', fid.read(4))[0],
                        'Var3': struct.unpack('f', fid.read(4))[0],
                        'Var4': struct.unpack('f', fid.read(4))[0],
                        'Var5': struct.unpack('f', fid.read(4))[0],
                        'Var6': struct.unpack('f', fid.read(4))[0],
                        'Var7': struct.unpack('f', fid.read(4))[0],
                        'Var8': struct.unpack('f', fid.read(4))[0],
                        'Var9': struct.unpack('f', fid.read(4))[0],
                        'Var10': struct.unpack('f', fid.read(4))[0],
                        'data': [] #list(struct.unpack(f'{data["groups"][x]["recs"][i]["npts"]}f', fid.read(4*data['groups'][x]['recs'][i]['npts'])))
                    }
                
                # skip all 10 cursors placeholders
                fid.seek(36*10, 1)
                record_data['data'] = list(struct.unpack(f'{record_data["npts"]}f', fid.read(4*record_data['npts'])))

                record_data['grp_d'] = datetime.utcfromtimestamp(record_data['grp_t'] / 86400 + datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')

                data['groups'][x]['recs'].append(record_data)

            if PLOT:
                import matplotlib.pyplot as plt

                # determine reasonable spacing between plots
                d = [x['data'] for x in data['groups'][x]['recs']]
                plot_offset = max(max(map(abs, [item for sublist in d for item in sublist]))) * 1.2

                plt.figure()

                for i in range(data['groups'][x]['nrecs']):
                    plt.plot([item - plot_offset * i for item in data['groups'][x]['recs'][i]['data']])
                    plt.hold(True)

                plt.title(f'Group {data["groups"][x]["grpn"]}')
                plt.axis('off')
                plt.show()

    return data

def get_str(data):
    # return string up until null character only
    ind = data.find(b'\x00')
    if ind > 0:
        data = data[:ind]
    return data.decode('utf-8')

def calculate_hearing_threshold(df, freq):
    db_values = []
    
    waves_array = []  # Array to store all waves

    for db in range(0,95,5):
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.loc[index, '0':]
            final = pd.to_numeric(final, errors='coerce')

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final
            db_values.append(db)

            waves_array.append(y_values.to_list())

    # Filter waves and dB values for the specified frequency
    waves_fd = FDataGrid(waves_array)
    fpca_discretized = FPCA(n_components=2)
    fpca_discretized.fit(waves_fd)
    projection = fpca_discretized.transform(waves_fd)

    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    neighbors = nearest_neighbors.fit(projection[:, :2])
    distances, indices = neighbors.kneighbors(projection[:, :2])
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    knee_locator = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    eps = distances[knee_locator.knee]

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps)
    clusters = dbscan.fit_predict(projection[:, :2])

    # Create DataFrame with projection results and cluster labels
    df = pd.DataFrame(projection[:, :2], columns=['1st_PC', '2nd_PC'])
    df['Cluster'] = clusters
    print(clusters)
    df['DB_Value'] = db_values

    # Find the minimum hearing threshold value among the outliers
    min_threshold = np.min(df[df['Cluster']==-1]['DB_Value'])

    return min_threshold

# Streamlit UI
st.title("Wave Plotting App")
st.sidebar.header("Upload File")
uploaded_files = st.sidebar.file_uploader("Choose a file", type=["csv", "arf"], accept_multiple_files=True)

annotations = []

if uploaded_files:
    dfs = []
    
    for file in uploaded_files:
        # Use tempfile
        temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.read())

        if file.name.endswith(".arf"):
        # Read ARF file
            data = arfread(temp_file.name)  
            
            # Process ARF data
            rows = []
            freqs = []
            dbs = []

            for group in data['groups']:
                for rec in group['recs']:
                    # Extract data
                    freq = rec['Var1']
                    db = rec['Var2']
                    
                    # Construct row  
                    wave_cols = list(enumerate(rec['data']))
                    wave_data = {f'{i}':v*1e6 for i, v in wave_cols} 
                    
                    row = {'Freq(Hz)': freq, 'Level(dB)': db, **wave_data}
                    rows.append(row)

            df = pd.DataFrame(rows)

        elif file.name.endswith(".csv"):
            # Process CSV
            if pd.read_csv(temp_file_path).shape[1] > 1:
                df = pd.read_csv(temp_file_path)
            else:
                df = pd.read_csv(temp_file_path, skiprows=2)
            
        # Append df to list
        dfs.append(df)

    # Get distinct frequency and dB level values across all files
    distinct_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in dfs]).unique())
    distinct_dbs = sorted(pd.concat([df['Level(dB)'] for df in dfs]).unique())
    

    multiply_y_factor = st.sidebar.number_input("Multiply Y Values by Factor", value=1.0)

    # Frequency dropdown options
    freq = st.sidebar.selectbox("Select Frequency (Hz)", options=distinct_freqs, index=0)

    # dB Level dropdown options
    db = st.sidebar.selectbox("Select dB Level", options=distinct_dbs, index=0)

    y_min = st.sidebar.slider("Y-axis Minimum", -2.5, -0.001, value = -2.5)
    y_max = st.sidebar.slider("Y-axis Maximum", 0.001, 2.5, value = 2.5)

    baseline_level_str = st.sidebar.text_input("Set Baseline Level", "0.0")
    baseline_level = float(baseline_level_str)

    plot_time_warped = st.sidebar.checkbox("Plot Time Warped Curves", False)

    # Create a plotly figure
    fig = go.Figure()

    #scatter_plot_option = st.sidebar.checkbox("Plot Waves as Scatter Plot", False)

    if st.sidebar.button("Plot Waves at Single Frequency"):
        if plot_time_warped:
            fig = plot_waves_single_frequency(df, freq, y_min, y_max, plot_time_warped=True)
        else:
            fig = plot_waves_single_frequency(df, freq, y_min, y_max, plot_time_warped=False)
        st.plotly_chart(fig)
        display_metrics_table_all_db(df, freq, distinct_dbs, baseline_level)

    if st.sidebar.button("Plot Waves at Single dB Level"):
        fig = plot_waves_single_db(df, db, y_min, y_max)
        st.plotly_chart(fig)
        display_metrics_table(df, freq, db, baseline_level)

    if st.sidebar.button("Plot Waves at Single Tuple (Frequency, dB)"):
        fig = plot_waves_single_tuple(df, freq, db, y_min, y_max)
        st.plotly_chart(fig)
        display_metrics_table(df, freq, db, baseline_level)
    
    if st.sidebar.button("Plot Stacked Waves at Single Frequency"):
        if plot_time_warped:
            fig = plot_waves_stacked(df, freq, y_min, y_max, plot_time_warped=True)
        else:
            fig = plot_waves_stacked(df, freq, y_min, y_max, plot_time_warped=False)
        st.plotly_chart(fig)
    
    #if st.sidebar.button("Plot Waves with Cubic Spline"):
    #    fig = plotting_waves_cubic_spline(df, freq, db)
    #    fig.update_layout(yaxis_range=[y_min, y_max])
    #    st.plotly_chart(fig)

    if st.sidebar.button("Plot 3D Surface"):
        fig_3d_surface = plot_3d_surface(df, freq, y_min, y_max)
        st.plotly_chart(fig_3d_surface)
    
    #st.markdown(get_download_link(fig), unsafe_allow_html=True)
