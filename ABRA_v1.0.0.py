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
import datetime
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.optim as optim
import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import cm
import colorcet as cc
import io
from numpy import AxisError
import warnings
warnings.filterwarnings('ignore')

# Co-authored by: Abhijeeth Erra and Jeffrey Chen

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 61, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 61)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def interpolate_and_smooth(final, target_length=244):
    if len(final) > target_length:
        new_points = np.linspace(0, len(final), target_length + 2)
        interpolated_values = np.interp(new_points, np.arange(len(final)), final)
        final = np.array(interpolated_values[:target_length], dtype=float)
    elif len(final) < target_length:
        st.write(final)
        original_indices = np.arange(len(final))
        target_indices = np.linspace(0, len(final) - 1, target_length)
        cs = CubicSpline(original_indices, final)
        final = cs(target_indices)
    return final

def plot_wave(fig, x_values, y_values, color, name, marker_color=None):
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=name, line=dict(color=color)))
    if marker_color:
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', marker=dict(color=marker_color), name=name, showlegend=False))

def calculate_and_plot_wave(df, freq, db, color, threshold=None):
    db_column = 'Level(dB)' if level else 'PostAtten(dB)'
    khz = df[(df['Freq(Hz)'] == freq) & (df[db_column] == db)]
    if not khz.empty:
        index = khz.index.values[0]
        final = df.loc[index, '0':].dropna()
        final = pd.to_numeric(final, errors='coerce').dropna()

        target = int(244 * (time_scale / 10))
        
        y_values = interpolate_and_smooth(final, target)  # Original y-values for plotting
        sampling_rate = len(y_values) / time_scale

        x_values = np.linspace(0, len(y_values) / sampling_rate, len(y_values))

        y_values_for_peak_finding = interpolate_and_smooth(final[:244])
        if units == 'Nanovolts':
            y_values /= 1000

        y_values_for_peak_finding *= multiply_y_factor

        highest_peaks, relevant_troughs = peak_finding(y_values_for_peak_finding)

        return x_values, y_values, highest_peaks, relevant_troughs
    return None, None, None, None

def plot_waves_single_frequency(df, freq, y_min, y_max, plot_time_warped=False):
    db_column = 'Level(dB)' if level else 'PostAtten(dB)'

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    for idx, file_df in enumerate(selected_dfs):
        fig = go.Figure()

        df_filtered = file_df[file_df['Freq(Hz)'] == freq]
        db_levels = sorted(df_filtered[db_column].unique())
        glasbey_colors = cc.glasbey[:len(db_levels)]

        original_waves = []

        try:
            threshold = np.abs(calculate_hearing_threshold(file_df, freq))
        except Exception as e:
            threshold = None
            st.write("Threshold can't be calculated.", e)

        for i, db in enumerate(sorted(db_levels)):
            if db_column == 'Level(dB)':
                x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, glasbey_colors[i])
            else:
                x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, glasbey_colors[i])
            
            if y_values is not None:
                if return_units == 'Nanovolts':
                    y_values *= 1000
                if db_column == 'Level(dB)':
                    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'{int(db)} dB', line=dict(color=glasbey_colors[i])))
                else:
                    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'{calibration_levels[(file_df.name, freq)] - int(db)} dB', line=dict(color=glasbey_colors[i])))

                # Mark the highest peaks with red markers
                fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks'))#, showlegend=False))

                # Mark the relevant troughs with blue markers
                fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs'))#, showlegend=False))

                if plot_time_warped:
                    original_waves.append(y_values.tolist())

        if plot_time_warped:
            original_waves_array = np.array([wave[:-1] for wave in original_waves])
            try:
                time = np.linspace(0, time_scale, original_waves_array.shape[1])
                obj = fs.fdawarp(original_waves_array.T, time)
                obj.srsf_align(parallel=True)
                warped_waves_array = obj.fn.T
                for i, db in enumerate(db_levels):
                    fig.add_trace(go.Scatter(x=np.linspace(0, 10, len(warped_waves_array[i])), y=warped_waves_array[i], mode='lines', name=f'{int(db)} dB', line=dict(color=glasbey_colors[i])))
            except IndexError:
                pass

        if threshold is not None:
            if db_column == 'Level(dB)':
                x_values, y_values, _, _ = calculate_and_plot_wave(file_df, freq, threshold, 'black')
            elif db_column == 'PostAtten(dB)':
                x_values, y_values, _, _ = calculate_and_plot_wave(file_df, freq, calibration_levels[(file_df.name, freq)] - threshold, 'black')
            if y_values is not None:
                if return_units == 'Nanovolts':
                    y_values *= 1000
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'Threshold: {int(threshold)} dB', line=dict(color='black', width=5)))
        
        if return_units == 'Nanovolts':
            y_units = 'Voltage (nV)'
        else:
            y_units = 'Voltage (μV)'

        fig.update_layout(title=f'{selected_files[idx].split("/")[-1]} - Frequency: {freq} Hz', xaxis_title='Time (ms)', yaxis_title=y_units)
        fig.update_layout(annotations=annotations)
        fig.update_layout(yaxis_range=[y_min, y_max])
        fig.update_layout(width=700, height=450)

        st.plotly_chart(fig)

def plot_waves_single_tuple(freq, db, y_min, y_max):
    fig = go.Figure()
    db_column = 'Level(dB)' if level else 'PostAtten(dB)'

    for idx, file_df in enumerate(selected_dfs):
        x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, 'blue')

        if y_values is not None:
            if return_units == 'Nanovolts':
                y_values *= 1000
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'{selected_files[idx].split("/")[-1]}', showlegend=False))
            # Mark the highest peaks with red markers
            fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks', showlegend=False))

            # Mark the relevant troughs with blue markers
            fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs', showlegend=False))

    if return_units == 'Nanovolts':
        y_units = 'Voltage (nV)'
    else:
        y_units = 'Voltage (μV)'

    fig.update_layout(width=700, height=450)
    fig.update_layout(xaxis_title='Time (ms)', yaxis_title=y_units, title=f'{selected_files[idx].split("/")[-1]}, Freq = {freq}, db SPL = {db}')
    fig.update_layout(annotations=annotations)
    fig.update_layout(yaxis_range=[y_min, y_max])
    fig.update_layout(font_family="Times New Roman",
                      font_color="black",
                      title_font_family="Times New Roman",
                      font=dict(size=18))

    return fig

def plot_3d_surface(df, freq, y_min, y_max):
    db_column = 'Level(dB)' if level else 'PostAtten(dB)'

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    for idx, file_df in enumerate(selected_dfs):
        fig = go.Figure()
        df_filtered = file_df[file_df['Freq(Hz)'] == freq]
        if db_column == 'Level(dB)':
            db_levels = sorted(df_filtered[db_column].unique(), reverse=True)
        else:
            db_levels = sorted([calibration_levels[(file_df.name, freq)] - db for db in df_filtered[db_column].unique()], reverse=True)
        
        original_waves = []

        try:
            threshold = calculate_hearing_threshold(file_df, freq)
        except:
            threshold = None

        for db in db_levels:
            if db_column == 'Level(dB)':
                x_values, y_values, _, _ = calculate_and_plot_wave(file_df, freq, db, 'blue')
            else:
                x_values, y_values, _, _ = calculate_and_plot_wave(file_df, freq, calibration_levels[(file_df.name, freq)] - db, 'blue')

            if y_values is not None:
                if return_units == 'Nanovolts':
                    y_values *= 1000
                original_waves.append(y_values.tolist())

        original_waves_array = np.array([wave[:-1] for wave in original_waves])

        try:
            time = np.linspace(0, 10, original_waves_array.shape[1])
            obj = fs.fdawarp(original_waves_array.T, time)
            obj.srsf_align(parallel=True)
            warped_waves_array = obj.fn.T
        except IndexError:
            warped_waves_array = np.array([])

        for i, (db, warped_waves) in enumerate(zip(db_levels, warped_waves_array)):
            fig.add_trace(go.Scatter3d(x=[db] * len(warped_waves), y=x_values, z=warped_waves, mode='lines', name=f'{int(db)} dB', line=dict(color='blue')))
            if db == threshold:
                fig.add_trace(go.Scatter3d(x=[db] * len(warped_waves), y=x_values, z=warped_waves, mode='lines', name=f'Thresh: {int(db)} dB', line=dict(color='black', width=5)))
                
        for i in range(len(time)):
            z_values_at_time = [warped_waves_array[j, i] for j in range(len(db_levels))]
            fig.add_trace(go.Scatter3d(x=db_levels, y=[time[i]] * len(db_levels), z=z_values_at_time, mode='lines', name=f'Time: {time[i]:.2f} ms', line=dict(color='rgba(0, 255, 0, 0.3)'), showlegend=False))

        fig.update_layout(width=700, height=450)
        fig.update_layout(title=f'{selected_files[idx].split("/")[-1]} - Frequency: {freq} Hz', scene=dict(xaxis_title='dB', yaxis_title='Time (ms)', zaxis_title='Voltage (μV)'), annotations=annotations)

        st.plotly_chart(fig)

def display_metrics_table(df, freq, db, baseline_level):
    if level:
        d = 'Level(dB)'
    else:
        d = 'PostAtten(dB)'

    khz = df[(df['Freq(Hz)'] == freq) & (df[d] == db)]
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

        highest_peaks, relevant_troughs = peak_finding(y_values)

        if highest_peaks.size > 0:  # Check if highest_peaks is not empty
            first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
            latency_to_first_peak = highest_peaks[0] * (10 / len(y_values))  # Assuming 10 ms duration for waveform

            if len(highest_peaks) >= 4:
                amplitude_ratio = (y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]) / (y_values[highest_peaks[3]] - y_values[relevant_troughs[3]])
            else:
                amplitude_ratio = np.nan

            metrics_table = pd.DataFrame({
                'Metric': ['First Peak Amplitude (μV)', 'Latency to First Peak (ms)', 'Amplitude Ratio (Peak1/Peak4)'],#, 'Estimated Threshold'],
                'Value': [first_peak_amplitude, latency_to_first_peak, amplitude_ratio]#, calculate_hearing_threshold(df, freq)],
            }).reset_index(drop=True)
            #st.table(metrics_table)
            styled_metrics_table = metrics_table.style.set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]},
                 {'selector': 'td', 'props': [('text-align', 'center')]}]
            ).set_properties(**{'width': '100px'})
        return styled_metrics_table

def display_metrics_table_all_db(selected_dfs, freqs, db_levels, baseline_level):
    if level:
        db_column = 'Level(dB)'
    else:
        db_column = 'PostAtten(dB)'

    ru = 'μV'
    if return_units == 'Nanovolts':
        ru = 'nV'
        
    metrics_data = {'File Name': [], 'Frequency (Hz)': [], 'dB Level': [], f'Wave I amplitude (P1-T1) ({ru})': [], 'Latency to First Peak (ms)': [], 'Amplitude Ratio (Peak1/Peak4)': [], 'Estimated Threshold': []}

    for file_df, file_name in zip(selected_dfs, selected_files):
        for freq in freqs:
            try:
                threshold = calculate_hearing_threshold(file_df, freq)
            except:
                threshold = np.nan
                pass

            for db in db_levels:
                _, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, 'blue')
                    
                if return_units == 'Nanovolts':
                    y_values *= 1000

                if highest_peaks is not None:
                    if highest_peaks.size > 0:  # Check if highest_peaks is not empty
                        first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                        latency_to_first_peak = highest_peaks[0] * (10 / len(y_values))  # Assuming 10 ms duration for waveform

                        if len(highest_peaks) >= 4 and len(relevant_troughs) >= 4:
                            amplitude_ratio = (y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]) / (
                                        y_values[highest_peaks[3]] - y_values[relevant_troughs[3]])
                        else:
                            amplitude_ratio = np.nan

                        metrics_data['File Name'].append(file_name.split("/")[-1])
                        metrics_data['Frequency (Hz)'].append(freq)
                        if db_column == 'Level(dB)':
                            metrics_data['dB Level'].append(db)
                        else:
                            metrics_data['dB Level'].append(calibration_levels[(df.name, freq)] - db)
                        metrics_data[f'Wave I amplitude (P1-T1) ({ru})'].append(first_peak_amplitude)
                        metrics_data['Latency to First Peak (ms)'].append(latency_to_first_peak)
                        metrics_data['Amplitude Ratio (Peak1/Peak4)'].append(amplitude_ratio)
                        metrics_data['Estimated Threshold'].append(threshold)

    metrics_table = pd.DataFrame(metrics_data)
    st.dataframe(metrics_table, hide_index=True, use_container_width=True)

def plot_waves_stacked(freq):
    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    db_column = 'Level(dB)' if level else 'PostAtten(dB)'

    for idx, file_df in enumerate(selected_dfs):
        fig = go.Figure()

        # Get unique dB levels and color palette
        unique_dbs = sorted(file_df[db_column].unique())
        num_dbs = len(unique_dbs)
        vertical_spacing = 25 / num_dbs
        db_offsets = {db: y_min + i * vertical_spacing for i, db in enumerate(unique_dbs)}
        glasbey_colors = cc.glasbey[:num_dbs]

        # Calculate the hearing threshold
        try:
            threshold = calculate_hearing_threshold(file_df, freq)
        except:
            threshold = None

        db_levels = sorted(unique_dbs, reverse=True) if db_column == 'Level(dB)' else sorted(unique_dbs)
        if db_column == 'PostAtten(dB)':
            db_levels = np.array(db_levels)
            calibration_level = np.full(len(db_levels), calibration_levels[(df.name, freq)])
            db_levels = calibration_level - db_levels
        max_db = db_levels[0]

        for i, db in enumerate(db_levels):
            try:
                if db_column == 'Level(dB)':
                    khz = file_df[(file_df['Freq(Hz)'] == freq) & (file_df[db_column] == db)]
                else:
                    khz = file_df[(file_df['Freq(Hz)'] == freq) & (file_df[db_column] == (calibration_level[0] - db))]

                if not khz.empty:
                    index = khz.index.values[-1]
                    final = file_df.loc[index, '0':].dropna()
                    final = pd.to_numeric(final, errors='coerce')
                    final = interpolate_and_smooth(final)
                    final *= multiply_y_factor

                    if units == 'Nanovolts':
                        final /= 1000

                    # Normalize the waveform
                    if (db_column == 'Level(dB)' and db == max_db) or (db_column == 'PostAtten(dB)' and db == max_db):
                        max_value = np.max(np.abs(final))
                    final_normalized = final / max_value

                    # Apply vertical offset
                    if db_column == 'Level(dB)':
                        y_values = final_normalized + db_offsets[db]
                    else:
                        y_values = final_normalized + db_offsets[calibration_level[0] - db]

                    # Plot the waveform
                    color_scale = glasbey_colors[i]
                    fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(y_values)),
                                            y=y_values,
                                            mode='lines',
                                            name=f'{int(db)} dB',
                                            line=dict(color=color_scale)))

                    if (db_column == 'Level(dB)' and db == threshold) or (db_column == 'PostAtten(dB)' and db == threshold):
                        fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(y_values)),
                                                y=y_values,
                                                mode='lines',
                                                name=f'Thresh: {int(db)} dB',
                                                line=dict(color='black', width=5),
                                                showlegend=True))
                    

                    fig.add_annotation(
                        x=10,
                        y=y_values[-1] + 0.5,
                        xref="x",
                        yref="y",
                        text=f"{int(db)} dB",
                        showarrow=False,
                        font=dict(size=10, color=color_scale),
                        xanchor="right"
                    )
            except Exception as e:
                st.write(f"Error processing dB level {db}: {e}")

        # Add vertical scale bar
        # if max_value and y_min >= -5 and y_min <= 1:
        #     scale_bar_length = 2 / max_value
        #     fig.add_trace(go.Scatter(x=[10.2, 10.2],
        #                              y=[0, scale_bar_length],
        #                              mode='lines',
        #                              line=dict(color='black', width=2),
        #                              showlegend=False))

        #     fig.add_annotation(
        #         x=10.3,
        #         y=scale_bar_length / 2,
        #         text=f"{2.0:.1f} μV",
        #         showarrow=False,
        #         font=dict(size=10, color='black'),
        #         xanchor="left",
        #         yanchor="middle"
        #     )

        fig.update_layout(title=f'{selected_files[idx].split("/")[-1]} - Frequency: {freq} Hz',
                          xaxis_title='Time (ms)',
                          yaxis_title='Voltage (μV)',
                          width=400,
                          height=700,
                          yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                          xaxis=dict(showgrid=False, zeroline=False))

        khz = file_df[(file_df['Freq(Hz)'] == freq)]
        if not khz.empty:
            st.plotly_chart(fig)

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
                #data['fileTime'] = datetime.datetime.utcfromtimestamp(ttt/86400 + datetime.datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')
                bFirstPass = False

            if isRZ:
                grp_t_format = 'q'
                beg_t_format = 'q'
                end_t_format = 'q'
                read_size = 8
            else:
                grp_t_format = 'I'
                beg_t_format = 'I'
                end_t_format = 'I'
                read_size = 4

            data['groups'][x]['beg_t'] = struct.unpack(beg_t_format, fid.read(read_size))[0]
            data['groups'][x]['end_t'] = struct.unpack(end_t_format, fid.read(read_size))[0]

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
                        'grp_t': struct.unpack(grp_t_format, fid.read(read_size))[0],
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
                        'beg_t': struct.unpack(beg_t_format, fid.read(read_size))[0],
                        'end_t': struct.unpack(end_t_format, fid.read(read_size))[0],
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

                #record_data['grp_d'] = datetime.datetime.utcfromtimestamp(record_data['grp_t'] / 86400 + datetime.datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')

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

def calculate_hearing_threshold(df, freq, baseline_level=100, multiply_y_factor=1):
    db_column = 'Level(dB)' if level else 'PostAtten(dB)'

    thresholding_model = load_model('models/abr_cnn_aug_norm_opt.keras')
    thresholding_model.steps_per_execution = 1
    
    # Filter DataFrame to include only data for the specified frequency
    df_filtered = df[df['Freq(Hz)'] == freq]

    # Get unique dB levels for the filtered DataFrame
    db_levels = sorted(df_filtered[db_column].unique(), reverse=True) if db_column == 'Level(dB)' else sorted(df_filtered[db_column].unique())
    waves = []

    for db in db_levels:
        khz = df_filtered[df_filtered[db_column] == np.abs(db)]
        if not khz.empty:
            index = khz.index.values[-1]
            final = df_filtered.loc[index, '0':].dropna()
            final = pd.to_numeric(final, errors='coerce')
            final = np.array(final, dtype=np.float64)
            target = int(244 * (time_scale / 10))
            y_values = interpolate_and_smooth(final, target)
            final = interpolate_and_smooth(final[:244])
            final *= multiply_y_factor

            if units == 'Nanovolts':
                final /= 1000

            waves.append(final)
    
    waves = np.array(waves)
    flattened_data = waves.flatten().reshape(-1, 1)
    scaler = StandardScaler()
    scaled_flattened_data = scaler.fit_transform(flattened_data).reshape(waves.shape)
    waves = np.expand_dims(scaled_flattened_data, axis=2)
    
    # Perform prediction
    prediction = thresholding_model.predict(waves)
    y_pred = (prediction > 0.5).astype(int).flatten()

    if db_column == 'PostAtten(dB)':
        db_levels = np.array(db_levels)
        calibration_level = np.full(len(db_levels), calibration_levels[(df.name, freq)])
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

    return lowest_db

def all_thresholds():
    df_dict = {'Filename': [],
               'Frequency': [],
               'Threshold': []}
    for (file_df, file_name) in zip(selected_dfs, selected_files):
        for hz in distinct_freqs:
            thresh = np.nan
            try:
                thresh = calculate_hearing_threshold(file_df, hz)
            except:
                pass
            df_dict['Filename'].append(file_name.split("/")[-1])
            df_dict['Frequency'].append(hz)
            df_dict['Threshold'].append(thresh)
    threshold_table = pd.DataFrame(df_dict)
    st.dataframe(threshold_table, hide_index=True, use_container_width=True)
    return threshold_table

def peak_finding(wave):
    # Prepare waveform
    waveform = interpolate_and_smooth(wave)
    waveform_torch = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    
    # Get prediction from model
    outputs = peak_finding_model(waveform_torch)
    prediction = int(round(outputs.detach().numpy()[0][0], 0))

    # Apply Gaussian smoothing
    smoothed_waveform = gaussian_filter1d(wave, sigma=1)

    # Find peaks and troughs
    n = 18
    t = 14
    start_point = prediction - 9
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

def calculate_unsupervised_threshold(df, freq):
    if level:
        db_column = 'Level(dB)'
    else:
        db_column = 'PostAtten(dB)'

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    waves_array = []  # Array to store all waves

    khz = df[(df['Freq(Hz)'] == freq)]
    db_values = sorted(khz[db_column].unique())
    for db in db_values:
        khz = df[(df['Freq(Hz)'] == freq) & (df[db_column] == db)]

        if not khz.empty:
            index = khz.index.values[-1]
            final = df.loc[index, '0':].dropna()
            final = pd.to_numeric(final, errors='coerce')

            if len(final) > 244:
                new_points = np.linspace(0, len(final), 245)
                interpolated_values = np.interp(new_points, np.arange(len(final)), final)
                interpolated_values = pd.Series(interpolated_values)
                final = np.array(interpolated_values[:244], dtype=float)
            if len(final) < 244:
                original_indices = np.arange(len(final))
                target_indices = np.linspace(0, len(final) - 1, 244)
                cs = CubicSpline(original_indices, final)
                smooth_amplitude = cs(target_indices)
                final = smooth_amplitude

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final
            
            if units == 'Nanovolts':
                y_values /= 1000

            waves_array.append(y_values.tolist())
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
    dfn = pd.DataFrame(projection[:, :2], columns=['1st_PC', '2nd_PC'])
    dfn['Cluster'] = clusters
    dfn['DB_Value'] = db_values

    # Find the minimum hearing threshold value among the outliers
    min_threshold = np.min(dfn[dfn['Cluster']==-1]['DB_Value'])

    return min_threshold

def plot_io_curve(df, freqs, db_levels, multiply_y_factor=1.0, units='Microvolts'):
    db_column = 'Level(dB)' if level else 'PostAtten(dB)'
    
    amplitudes = []

    ru = 'μV'
    if return_units == 'Nanovolts':
        ru = 'nV'

    for file_df, file_name in zip(selected_dfs, selected_files):
        for freq in freqs:
            try:
                threshold = calculate_hearing_threshold(file_df, freq)
            except:
                threshold = np.nan
                pass

            for db in db_levels:
                _, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, 'blue')
                    
                if return_units == 'Nanovolts':
                    y_values *= 1000

                if highest_peaks is not None:
                    if highest_peaks.size > 0:  # Check if highest_peaks is not empty
                        first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                        amplitudes.append(first_peak_amplitude)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.full(len(db_levels), calibration_levels[(file_df.name, freq)]) - db_levels, y=amplitudes, mode='lines+markers', name=f'Freq: {freq} Hz'))
    
    fig.update_layout(
        title=f'I/O Curve for Frequency {freq} Hz',
        xaxis_title='dB Level',
        yaxis_title=f'Wave 1 Amplitude ({ru})',
        xaxis=dict(tickmode='linear', dtick=5),
        yaxis=dict(range=[min(amplitudes) - 0.1 * abs(min(amplitudes)), max(amplitudes) + 0.1 * abs(max(amplitudes))]),
        template='plotly_white'
    )
    fig.update_layout(font_family="Times New Roman",
                      font_color="black",
                      title_font_family="Times New Roman",
                      font=dict(size=24))

    st.plotly_chart(fig)
    return fig

# Streamlit UI
st.title("Wave Plotting App")
st.sidebar.header("Upload File")
uploaded_files = st.sidebar.file_uploader("Choose a file", type=["csv", "arf"], accept_multiple_files=True)
is_rz_file = st.sidebar.radio("Select ARF File Type:", ("RP", "RZ"))
is_click = st.sidebar.radio("Click or Tone? (for .arf files)", ("Click", "Tone"))
click = None
if is_click == "Click":
    click = True
else:
    click = False
is_level = st.sidebar.radio("Select dB You Are Studying:", ("Level", "Attenuation"))

annotations = []

peak_finding_model = CNN()
model_loader = torch.load('./models/waveI_cnn_model.pth')
peak_finding_model.load_state_dict(model_loader)
peak_finding_model.eval()

if uploaded_files:
    dfs = []
    selected_files = []
    selected_dfs = []
    calibration_levels = {}

    
    st.sidebar.write("Select files to analyze:")
    for idx, file in enumerate(uploaded_files):
        # Use tempfile
        temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.read())
        #st.sidebar.markdown(f"**File Name:** {file.name}")
        selected = st.sidebar.checkbox(f"{file.name}", key=f"file_{idx}")
        
        if selected:
            selected_files.append(temp_file_path)

        if file.name.endswith(".arf"):
        # Read ARF file
            if is_rz_file == 'RP':
                data = arfread(temp_file.name, RP=True) 
            else:
                data = arfread(temp_file.name) 
            
            # Process ARF data
            rows = []
            freqs = []
            dbs = []

            for group in data['groups']:
                for rec in group['recs']:
                    # Extract data
                    if not click:
                        freq = rec['Var1']
                        db = rec['Var2']
                    else:
                        freq = 'Click'
                        db = rec['Var1']
                    
                    # Construct row  
                    wave_cols = list(enumerate(rec['data']))
                    wave_data = {f'{i}':v*1e6 for i, v in wave_cols} 
                    
                    if is_level == 'Level':
                        row = {'Freq(Hz)': freq, 'Level(dB)': db, **wave_data}
                        rows.append(row)
                    if is_level == 'Attenuation':
                        row = {'Freq(Hz)': freq, 'PostAtten(dB)': db, **wave_data}
                        rows.append(row)

            df = pd.DataFrame(rows)

        elif file.name.endswith(".csv"):
            # Process CSV
            if pd.read_csv(temp_file_path).shape[1] > 1:
                df = pd.read_csv(temp_file_path)
            else:
                df = pd.read_csv(temp_file_path, skiprows=2)
            
        # Append df to list
        df.name = file.name
        dfs.append(df)
        if temp_file_path in selected_files:
            selected_dfs.append(df)

    level = (is_level == 'Level')

    db_column = 'Level(dB)' if level else 'PostAtten(dB)'

    # Get distinct frequency and dB level values across all files
    distinct_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in dfs]).unique())
    distinct_dbs = sorted(pd.concat([df['Level(dB)'] if level else df['PostAtten(dB)'] for df in dfs]).unique())

    time_scale = st.sidebar.number_input("Time Scale for Recording (ms)", value=10.0)
    
    multiply_y_factor = st.sidebar.number_input("Multiply Y Values by Factor", value=1.0)

    # Unit dropdown options
    units = st.sidebar.selectbox("Select Units Used in Collecting Your Data", options=['Microvolts', 'Nanovolts'], index=0)

    # Unit dropdown options
    return_units = st.sidebar.selectbox("Select Units You Would Like to Analyze With", options=['Microvolts', 'Nanovolts'], index=0)

    # Frequency dropdown options
    freq = st.sidebar.selectbox("Select Frequency (Hz)", options=distinct_freqs, index=0)

    # dB Level dropdown options
    db = st.sidebar.selectbox(f'Select dB {is_level}', options=distinct_dbs, index=0)

    if return_units == 'Nanovolts':
        ymin = -5000.0
        ymax = 5000.0
    else:
        ymin = -5.0
        ymax = 5.0

    y_min = st.sidebar.number_input("Y-axis Minimum", value=ymin)
    y_max = st.sidebar.number_input("Y-axis Maximum", value=ymax)

    baseline_level_str = st.sidebar.text_input("Set Baseline Level", "0.0")
    baseline_level = float(baseline_level_str)

    plot_time_warped = st.sidebar.checkbox("Plot Time Warped Curves", False)

    if not level:
        st.sidebar.subheader("Calibration Levels")
        for file in selected_files:
            for hz in distinct_freqs:
                key = (os.path.basename(file), hz)
                calibration_levels[key] = st.sidebar.number_input(f"Calibration Level for {os.path.basename(file)} at {hz} Hz", value=0.0)

    # Create a plotly figure
    fig = go.Figure()
    
    if st.sidebar.button("Plot Waves at Single Frequency"):
        if plot_time_warped:
            fig = plot_waves_single_frequency(df, freq, y_min, y_max, plot_time_warped=True)
        else:
            fig = plot_waves_single_frequency(df, freq, y_min, y_max, plot_time_warped=False)
        display_metrics_table_all_db(selected_dfs, [freq], distinct_dbs, baseline_level)

    if st.sidebar.button("Plot Single Wave (Frequency, dB)"):
        fig = plot_waves_single_tuple(freq, db, y_min, y_max)
        st.plotly_chart(fig)
        fig.write_image("fig1.pdf")
        display_metrics_table_all_db(selected_dfs, [freq], [db], baseline_level)
        # Create an in-memory buffer
        buffer = io.BytesIO()

        # Save the figure as a pdf to the buffer
        fig.write_image(file=buffer, format="pdf")

        # Download the pdf from the buffer
        st.download_button(
            label="Download PDF",
            data=buffer,
            file_name="figure.pdf",
            mime="application/pdf",
        )
    
    if st.sidebar.button("Plot Stacked Waves at Single Frequency"):
        plot_waves_stacked(freq)
    
    #if st.sidebar.button("Plot Waves with Cubic Spline"):
    #    fig = plotting_waves_cubic_spline(df, freq, db)
    #    fig.update_layout(yaxis_range=[y_min, y_max])
    #    st.plotly_chart(fig)

    if st.sidebar.button("Plot 3D Surface"):
        plot_3d_surface(df, freq, y_min, y_max)

    if st.sidebar.button("Plot I/O Curve"):
        fig = plot_io_curve(df, [freq], distinct_dbs)
        # Create an in-memory buffer
        buffer = io.BytesIO()

        # Save the figure as a pdf to the buffer
        fig.write_image(file=buffer, format="pdf")

        # Download the pdf from the buffer
        st.download_button(
            label="Download PDF",
            data=buffer,
            file_name="figure.pdf",
            mime="application/pdf",
        )
    
    if st.sidebar.button("Return All Thresholds"):
        all_thresholds()
    
    if st.sidebar.button("Return All Peak Analyses"):
        display_metrics_table_all_db(selected_dfs, distinct_freqs, distinct_dbs, baseline_level)
    
    #if st.sidebar.button("Plot Waves with Gaussian Smoothing"):
    #    fig_gauss = plotting_waves_gauss(dfs, freq, db)
    #    st.plotly_chart(fig_gauss)
    
    #st.markdown(get_download_link(fig), unsafe_allow_html=True)
