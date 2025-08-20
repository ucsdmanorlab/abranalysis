import datetime
import io
import os
import colorcet as cc
import fdasrsf as fs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.cluster import DBSCAN
import streamlit as st

from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from utils.calculate import *
from utils.plotting import apply_units, style_layout
from utils.processFiles import get_selected_data, process_uploaded_files_cached
import warnings
warnings.filterwarnings('ignore')

# TODO: use session states so plots don't disappear when downloading files
# TODO: consider converting freqs to kHz throughout for readability
# TODO: correct units
# TODO: make 3D plots work for tsv files
# TODO: check timescale of tsvs in training data...

# Co-authored by: Abhijeeth Erra and Jeffrey Chen

def check_settings_and_clear_cache():
    calc_settings = {
        'time_scale': st.session_state.get('time_scale', 10.0),
        'multiply_y_factor': st.session_state.get('multiply_y_factor', 1.0),
        'units': st.session_state.get('units', 'Microvolts'),
        'smooth_on': st.session_state.get('smooth_on', True),
        'return_units': st.session_state.get('return_units', 'Microvolts'),
        'calibration_levels': dict(st.session_state.get('calibration_levels', {}))
    }
    
    # Check if settings changed
    if 'previous_calc_settings' in st.session_state:
        if st.session_state.previous_calc_settings != calc_settings:
            if 'calculated_thresholds' in st.session_state:
                st.session_state.calculated_thresholds.clear()
            if 'calculated_waves' in st.session_state:
                st.session_state.calculated_waves.clear()
    
    st.session_state.previous_calc_settings = calc_settings

def plot_waves_single_dB(selected_dfs, selected_files, db, plot_time_warped=False, show_peaks=True):
    db_column = db_column_name()
    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return
    
    fig_list = []
    not_found_list = [] 
    for idx, file_df in enumerate(selected_dfs):
        if db not in file_df[db_column].unique():
            not_found_list.append(selected_files[idx].split('/')[-1])
            continue
        fig = go.Figure()
        df_filtered = file_df[file_df[db_column] == db]
        freqs = sorted(df_filtered['Freq(Hz)'].unique())

        glasbey_colors = cc.glasbey[:len(freqs)]

        original_waves = []

        for i, freq in enumerate(sorted(freqs)):
            x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db)

            if y_values is not None:
                y_values = apply_units(y_values)
                
                if plot_time_warped:
                    original_waves.append(y_values.tolist())
                    continue

                color = glasbey_colors[i]
                width = 2
                name = freq if type(freq)==str else f'{int(freq)} Hz'
                
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=name, line=dict(color=color, width=width), showlegend=st.session_state.show_legend))
                
                if show_peaks:
                    # Mark the highest peaks with red markers
                    fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks', showlegend=st.session_state.show_legend))

                    # Mark the relevant troughs with blue markers
                    fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs', showlegend=st.session_state.show_legend))

        if plot_time_warped:
            original_waves_array = np.array([wave[:-1] for wave in original_waves])
            try:
                time = np.linspace(0, st.session_state.time_scale, original_waves_array.shape[1])
                obj = fs.fdawarp(original_waves_array.T, time)
                obj.srsf_align(parallel=True)
                warped_waves_array = obj.fn.T
                for i, freq in enumerate(freqs):
                    color = glasbey_colors[i]
                    width = 2
                    name = freq if type(freq)==str else f'{int(freq)} Hz'
                    fig.add_trace(go.Scatter(x=np.linspace(0, st.session_state.time_scale, len(warped_waves_array[i])), y=warped_waves_array[i], mode='lines', name=name, line=dict(color=color, width=width), showlegend=st.session_state.show_legend))
            except IndexError:
                pass

        fig = style_layout(fig,
                           f'{selected_files[idx].split("/")[-1]} - {db} dB'
                           )                           
        fig_list.append(fig)
        #except Exception as e:
        #        st.write(f"Error processing freq {freq}: for file {selected_files[idx]} {e}")
    if len(not_found_list) > 0:
        st.write(f"{db} dB not found in files: {', '.join(not_found_list)}")
    return fig_list

def plot_waves_single_frequency(selected_dfs, selected_files, freq, plot_time_warped=False, show_peaks=True):
    db_column = db_column_name()

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return
    
    fig_list = []
    not_found_list = []
    for idx, file_df in enumerate(selected_dfs):
        # check if frequency exists in df:
        if freq not in file_df['Freq(Hz)'].unique():
            not_found_list.append(selected_files[idx].split('/')[-1])
            continue
        #try:
        fig = go.Figure()
        df_filtered = file_df[file_df['Freq(Hz)'] == freq]
        db_levels = sorted(df_filtered[db_column].unique())
        glasbey_colors = cc.glasbey[:len(db_levels)]

        original_waves = []

        try:
            threshold = np.abs(calculate_hearing_threshold(file_df, freq))
        except Exception as e:
            threshold = None
         
        for i, db in enumerate(sorted(db_levels)):
            x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db)
            
            if y_values is not None:
                y_values = apply_units(y_values)
                if plot_time_warped:
                    original_waves.append(y_values.tolist())
                    continue

                color = 'black' if db == threshold else glasbey_colors[i]
                width = 5 if db == threshold else 2
                name = f'{int(db_value(file_df.name, freq, db))} dB' 
                if db == threshold:
                    name = 'Threshold: ' + name
                
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=name, line=dict(color=color, width=width)))

                if show_peaks:
                    # Mark the highest peaks with red markers
                    fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks', showlegend=st.session_state.show_legend))

                    # Mark the relevant troughs with blue markers
                    fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs', showlegend=st.session_state.show_legend))

        if plot_time_warped:
            original_waves_array = np.array([wave[:-1] for wave in original_waves])
            try:
                time = np.linspace(0, st.session_state.time_scale, original_waves_array.shape[1])
                obj = fs.fdawarp(original_waves_array.T, time)
                obj.srsf_align(parallel=True)
                warped_waves_array = obj.fn.T
                for i, db in enumerate(db_levels):
                    color = 'black' if db == threshold else glasbey_colors[i]
                    width = 5 if db == threshold else 2
                    name = f'{int(db_value(file_df.name, freq, db))} dB'
                    if db == threshold:
                        name = 'Threshold: ' + name
                    fig.add_trace(go.Scatter(x=np.linspace(0, st.session_state.time_scale, len(warped_waves_array[i])), y=warped_waves_array[i], mode='lines', name=name, line=dict(color=color, width=width)))
            except IndexError:
                pass
        
        fig = style_layout(fig,
                           f'{selected_files[idx].split("/")[-1]} - Frequency: {freq} Hz'
                           )
        fig_list.append(fig)
        #except Exception as e:
        #        st.write(f"Error processing freq {freq}: for file {selected_files[idx]} {e}")
    if len(not_found_list) > 0:
        st.write(f"Frequency {freq} not found in files: {', '.join(not_found_list)}")
    return fig_list

def plot_waves_single_tuple(selected_dfs, selected_files, freq, db, show_peaks=True):
    fig = go.Figure()
    file_list = []
    for idx, file_df in enumerate(selected_dfs):
        x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db)
        if y_values is not None:
            y_values = apply_units(y_values)
            file_list.append(selected_files[idx].split("/")[-1])
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'{selected_files[idx].split("/")[-1]}'))#, showlegend=False))
            if show_peaks:
                # Mark the highest peaks with red markers
                fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks'))#, showlegend=False))

                # Mark the relevant troughs with blue markers
                fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs'))#, showlegend=False))
    if len(file_list) == 0:
        st.write("No files selected or no data available for the specified frequency and dB.")
        return
    db_str = f'{db_value(file_df.name, freq, db)}'
    fi_str = f'{file_list[0]}' if len(file_list) == 1 else f'{file_list[0]} and {len(file_list) - 1} more'
    figtitle = f'{fi_str}, {freq} Hz, {db_str} dB SPL' 
    fig = style_layout(fig,
                       figtitle
                       )
    return fig

def plot_3d_surface(selected_dfs, selected_files, freq):
    db_column = db_column_name()

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    fig_list = []
    for idx, file_df in enumerate(selected_dfs):
        fig = go.Figure()
        df_filtered = file_df[file_df['Freq(Hz)'] == freq]

        db_levels = df_filtered[db_column].unique()
        db_levels = sorted([db_value(file_df.name, freq, db) for db in db_levels], reverse=True)
        original_waves = []

        try:
            threshold = calculate_hearing_threshold(file_df, freq)
        except:
            threshold = None

        for db in db_levels:
            x_values, y_values, _, _ = calculate_and_plot_wave(file_df, freq, db_value(file_df.name, freq, db))
            
            if y_values is not None:
                y_values = apply_units(y_values)
                original_waves.append(y_values.tolist())

        original_waves_array = np.array([wave[:-1] for wave in original_waves])

        try:
            time = np.linspace(0, st.session_state.time_scale, original_waves_array.shape[1])
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

        fig.update_layout(width=700, height=600)
        fig.update_layout(title=f'{selected_files[idx].split("/")[-1]} - Frequency: {freq} Hz', scene=dict(xaxis_title='dB', yaxis_title='Time (ms)', zaxis_title='Voltage (μV)'),)
        camera = dict(
            up=dict(x=0, y=0, z=0.5),
            center=dict(x=0, y=0, z=-.1),
            eye=dict(x=1.3, y=1.3, z=1.3)
        )

        fig.update_layout(scene_camera=camera)
        fig.update_layout(font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
                      font_color="black",
                      title_font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
                      font=dict(size=14))
        fig.update_layout(showlegend=st.session_state.show_legend)
        fig_list.append(fig)
    return fig_list

def plot_waves_stacked(selected_dfs, selected_files, freq, stacked_labels=None):
    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return
    atten = st.session_state.get('atten', False)
    db_column = db_column_name()

    fig_list = []
    not_found_list = []
    for idx, file_df in enumerate(selected_dfs):
        fig = go.Figure()
        if freq not in file_df['Freq(Hz)'].unique():
            not_found_list.append(selected_files[idx].split('/')[-1])
            continue
        # Get unique dB levels and color palette
        df_filtered = file_df[file_df['Freq(Hz)'] == freq]
        unique_dbs = sorted(df_filtered[db_column].unique())
        if atten:
            unique_dbs = sorted(unique_dbs, reverse=True)
        num_dbs = len(unique_dbs)
        vertical_spacing = st.session_state.vert_space / num_dbs
        # Get y_min from input_settings if available, else default to 0.0
        # y_min = input_settings.y_min if input_settings and 'y_min' in input_settings else 0.0
        db_offsets = {db: i * vertical_spacing for i, db in enumerate(unique_dbs)}
        glasbey_colors = cc.glasbey[:num_dbs]

        # Calculate the hearing threshold
        try:
            threshold = calculate_hearing_threshold(file_df, freq)
        except:
            threshold = None

        db_levels = sorted(unique_dbs, reverse=True) if db_column == 'Level(dB)' else sorted(unique_dbs)
        if db_column == 'PostAtten(dB)':
            db_levels = np.array(db_levels)
            calibration_level = np.full(len(db_levels), st.session_state.calibration_levels[(file_df.name, freq)])
            db_levels = calibration_level - db_levels

        # db_levels = sorted([db_value(file_df.name, freq, db) for db in unique_dbs], reverse=True)
        
        max_db = db_levels[0]

        for i, db in enumerate(db_levels):
            try:
                is_thresh = False
                if db_column == 'Level(dB)':
                    khz = file_df[(file_df['Freq(Hz)'] == freq) & (file_df[db_column] == db)]
                else:
                    khz = file_df[(file_df['Freq(Hz)'] == freq) & (file_df[db_column] == (calibration_level[0] - db))]

                if not khz.empty:
                    index = khz.index.values[-1]
                    final = file_df.loc[index, '0':].dropna()
                    final = pd.to_numeric(final, errors='coerce')
                    final = interpolate_and_smooth(final)
                    final *= st.session_state.multiply_y_factor

                    # if units == 'Nanovolts':
                    #     final /= 1000

                    # Normalize the waveform
                    if (db == max_db):
                        max_value = np.max(np.abs(final))
                    final_normalized = final / max_value

                    # Apply vertical offset
                    if db_column == 'Level(dB)':
                        y_values = final_normalized + db_offsets[db]
                    else:
                        y_values = final_normalized + db_offsets[calibration_level[0] - db]

                    # Plot the waveform
                    color_scale = glasbey_colors[i]
                    fig.add_trace(go.Scatter(x=np.linspace(0, st.session_state.time_scale, len(y_values)),
                                            y=y_values,
                                            mode='lines',
                                            name=f'{int(db)} dB',
                                            line=dict(color=color_scale)))

                    if (db_column == 'Level(dB)' and db == threshold) or (db_column == 'PostAtten(dB)' and calibration_level[0] - db == threshold):
                        is_thresh=True
                        fig.add_trace(go.Scatter(x=np.linspace(0, st.session_state.time_scale, len(y_values)),
                                                y=y_values,
                                                mode='lines',
                                                name=f'Thresh: {int(db)} dB',
                                                line=dict(color='black', width=5),
                                                #showlegend=True
                                                ))
                        
                    y_pos = db_offsets[db] if not atten else db_offsets[calibration_level[0] - db]
                    if stacked_labels=="Left outside":
                        x_pos = 0; 
                    elif stacked_labels=="Right outside":
                        x_pos = st.session_state.time_scale*1.1; 
                    elif stacked_labels=="Right inside":
                        x_pos = st.session_state.time_scale; y_pos = y_pos + st.session_state.vert_space/num_dbs/3 #y_values.iloc[-1]
                    else:
                        continue
                    
                    fig.add_annotation(
                        x=x_pos,
                        y=y_pos,
                        xref="x",
                        yref="y",
                        text=f"<b>{int(db)} dB</b>" if is_thresh else f"{int(db)} dB",
                        showarrow=False,
                        font=dict(size=18, color='black' if is_thresh else color_scale),
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
        fig = style_layout(fig,
                           f'{selected_files[idx].split("/")[-1]} - Frequency: {freq} Hz',
        )
        fig.update_layout(width=400, height=700,
                          yaxis_title='Voltage (μV)',
                          yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                          xaxis=dict(showgrid=True, zeroline=False))

        khz = file_df[(file_df['Freq(Hz)'] == freq)]
        if not khz.empty:
            fig_list.append(fig)
    if len(not_found_list) > 0:
        st.write(f"Frequency {freq} not found in files: {', '.join(not_found_list)}")
    return fig_list

def all_thresholds(selected_dfs, selected_files, distinct_freqs):
    progress_bar = st.progress(0)
    status_text = st.empty()
    df_dict = {'Filename': [],
               'Frequency': [],
               'Threshold': []}
    for (file_df, file_name) in zip(selected_dfs, selected_files):
        status_text.text(f"{file_name.split('/')[-1]}")
        progress_bar.progress((selected_files.index(file_name) + 1) / len(selected_files))
        for hz in distinct_freqs:
            if hz not in file_df['Freq(Hz)'].unique():
                continue
            # df_filtered = file_df[(file_df['Freq(Hz)'] == hz)]
            # if df_filtered.empty:
            #     continue
            try:
                thresh = calculate_hearing_threshold(file_df, hz)
            except:
                thresh = np.nan
                pass
            df_dict['Filename'].append(file_name.split("/")[-1])
            df_dict['Frequency'].append(hz)
            df_dict['Threshold'].append(thresh)
    threshold_table = pd.DataFrame(df_dict)
    progress_bar.empty()
    status_text.empty()
    return threshold_table



def calculate_unsupervised_threshold(df, freq): # UNUSED
    db_column = db_column_name()

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

            final = final * multiply_y_factor
            if st.session_state.units == 'Nanovolts':
                final /= 1000

            final = interpolate_and_smooth(final)

            y_values = final

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

def db_column_name():
    atten = st.session_state.get('atten', False)
    return 'Level(dB)' if not atten else 'PostAtten(dB)'

def db_value(file_name, freq, db):
    atten = st.session_state.get('atten', False)
    if atten:
        return st.session_state.calibration_levels[(file_name, freq)] - int(db)
    else:
        return db

def plot_io_curve(selected_dfs, selected_files, freqs, db_levels):

    amplitudes = {}

    ru = 'μV'
    if st.session_state.return_units == 'Nanovolts':
        ru = 'nV'

    fig_list = []

    for file_df, file_name in zip(selected_dfs, selected_files):
        fig = go.Figure()
        include_fig = False
        for freq in freqs:
            # check if freq, file pair exists:
            if freq not in file_df['Freq(Hz)'].unique():
                continue
            db_levels_cal = [db_value(file_df.name, freq, db) for db in db_levels]
            for i, db in enumerate(db_levels):
                _, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db)
                if highest_peaks is None:
                    continue
                else:
                    if highest_peaks.size > 0: 
                        include_fig = True
                        y_values = apply_units(y_values)
                        first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                        amplitudes[db_levels_cal[i]] = first_peak_amplitude
                        
            if include_fig:
                x_vals = sorted(list(amplitudes.keys()))
                y_vals = [amplitudes[x] for x in x_vals]  # Get values in same order as x_vals
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=f'Freq: {freq} Hz', showlegend=st.session_state.show_legend))
        if include_fig:  
            file_name = file_name.split('/')[-1]
            freq_str = f'{freq} Hz' if len(freqs) == 1 else "all frequencies"
            fig.update_layout(
                title=f'{file_name} I/O Curve for {freq_str}',
                xaxis_title='dB SPL',
                yaxis_title=f'Wave 1 Amplitude ({ru})',
                xaxis=dict(tickmode='linear', dtick=5),
                yaxis=dict(range=[0, max(amplitudes.values()) + 0.1 * abs(max(amplitudes.values()))]),
                template='plotly_white'
            )
            fig.update_layout(font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
                            font_color="black",
                            title_font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
                            font=dict(size=24))

            fig_list.append(fig)
    return fig_list

def main():    
    if 'calibration_levels' not in st.session_state:
        st.session_state.calibration_levels = {}

    # Streamlit UI
    st.title("ABRA")
    tab1, tab2 = st.sidebar.tabs(["Data", "Plotting and Analysis"])
    
    uploaded_files = tab1.file_uploader("**Upload files to analyze:**", type=["csv", "arf", "asc", "tsv"], accept_multiple_files=True)
    is_rz_file = "RZ"

    # Inputs:
    inputs = tab1.expander("Input data properties", expanded=True)
    placeholder = inputs.empty()
    units = inputs.selectbox("Units used in collection", options=['Microvolts', 'Nanovolts'], index=0, key="units")
    # baseline_level_str = inputs.text_input("Set Baseline Level", "0.0")
    # baseline_level = float(baseline_level_str)
    is_click = inputs.radio("Tone or click? (for .arf files)", ("Tone", "Click"), horizontal=True)
    click = True if is_click == "Click" else False
    is_atten = inputs.toggle("dB saved as attenuation (.arf only)", value=False, key="atten")

    if uploaded_files:

        dfs, duration = process_uploaded_files_cached(uploaded_files, is_rz_file, click, is_atten)

        if duration is not None:
            time_scale = placeholder.number_input("Time scale of recording (detected from file, ms)", value=duration, format="%0.6f", key="time_scale")
        else:
            time_scale = placeholder.number_input("Time scale of recording (ms)", value=10.0, key="time_scale")

        tab2.write("**Select files to analyze:**")
        for idx, df in enumerate(dfs):
            tab2.checkbox(f"{df.name}", key=f"file_{idx}", value=True)
            
        selected_files, selected_dfs = get_selected_data()

        level = (not is_atten)
        db_column = db_column_name()

        if dfs:
            distinct_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in dfs]).unique())
            distinct_dbs = sorted(pd.concat([df[db_column_name()] for df in dfs]).unique())
        else:
            distinct_freqs, distinct_dbs = [], []

        if is_atten:
            cal_levels = tab1.expander("Calibration dB levels", expanded=True)
            # TODO: ask if this needs to be set per file or should be generalized across all files??
            for file_path in selected_files: 
                file_name = os.path.basename(file_path)
                for hz in distinct_freqs:
                    key = (file_name, hz)
                    if key not in st.session_state.calibration_levels:
                        st.session_state.calibration_levels[key] = 100.0
                    st.session_state.calibration_levels[key] = cal_levels.number_input(
                        f"Calibration dB for {file_name} at {hz} Hz",
                        value=st.session_state.calibration_levels[key],
                        step=5.0,
                        format="%0.1f",
                        key=f"cal_{file_name}_{hz}"
                    )
            # TODO: add option to set legend to show attenuation
            #atten_legend = cal_levels.toggle("Use attenuation levels in plot legends", value=False)

        # Output settings:
        outputs = tab2.expander("Output and plot settings", expanded=False)
        return_units = outputs.selectbox("Units for plots and outputs", options=['Microvolts', 'Nanovolts'], index=0, key="return_units")
        if return_units == 'Nanovolts':
            ymin = -5000.0
            ymax = 5000.0
        else:
            ymin = -5.0
            ymax = 5.0
        auto_y = outputs.toggle("Auto Y-axis scaling", value=True, key="auto_y")
        y_min = outputs.number_input("Y-axis minimum", value=ymin, disabled=auto_y, key="y_min")
        y_max = outputs.number_input("Y-axis maximum", value=ymax, disabled=auto_y, key="y_max")
        plot_time_warped = outputs.toggle("Plot time warped curves", False, key="plot_time_warped")
        show_legend = outputs.toggle("Show legend", True, key="show_legend")
        show_peaks = outputs.toggle("Show peaks (single wave and single frequency plots)", True)
        serif_font = outputs.toggle("Use serif fonts in plots", value=False, key="serif_font")

        advanced_settings = tab2.expander("Advanced settings", expanded=False)
        multiply_y_factor = advanced_settings.number_input("Multiply Y values by factor", value=1.0, key="multiply_y_factor")
        vert_space = advanced_settings.number_input("Vertical space (for stacked curves)", value=10.0, min_value=0.0, step=1.0, key="vert_space")
        stacked_labels = advanced_settings.selectbox("Stacked labels position", options=["Left outside", "Right outside", "Right inside", "Off"], index=2)
        all_peaks = advanced_settings.toggle("Output all peaks and troughs (experimental)", value=False, key="all_peaks")
        smooth_on = advanced_settings.toggle("Smooth wave", value=True, key="smooth_on")
        #unfilter_pk1 = advanced_settings.toggle("Include sub-threshold peak 1 amplitudes", value=False, key="unfilter_pk1")

        check_settings_and_clear_cache()

        # Frequency dropdown options
        allowed_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in selected_dfs]).unique())
        freq = tab2.selectbox("Select frequency (Hz)", options=allowed_freqs, index=0)
        # dB Level dropdown options, default to last (highest) dB)
        allowed_dbs = sorted(pd.concat([df[df['Freq(Hz)']==freq][db_column_name()] for df in selected_dfs]).unique())
        db = tab2.selectbox("Select sound level (dB)" if level else "Select sound level (attenuation, dB)", 
                            options=allowed_dbs, 
                            index=len(allowed_dbs)-1 if len(allowed_dbs) > 0 and level else 0)
        
        # Create a plotly figure
        #fig = go.Figure()
        tab2.header("Plot:")
        freq_str = freq if type(freq) == str else str(freq/1000) + " kHz"
        db_str = str(int(db)) + " dB"
        if tab2.button("Single wave ("+freq_str+", "+db_str+")", use_container_width=True):
            fig = plot_waves_single_tuple(selected_dfs, selected_files,freq, db, show_peaks=show_peaks)
            st.session_state['current_plots'] = [fig]
            st.session_state['current_plot_filenames'] = [(selected_files[0].split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_"+db_str.replace(' ','')+".pdf") if len(selected_files)==1 else "all_files_" +freq_str.replace(' ','')+ "_"+db_str.replace(' ','')+".pdf"]
            st.session_state['current_table'] =  display_metrics_table_all_db(selected_dfs, selected_files, [freq], [db])
        
        freqbuttons1, freqbuttons2 = tab2.columns([1, 1.5])
        if freqbuttons1.button("Single frequency", use_container_width=True):
            fig_list = plot_waves_single_frequency(selected_dfs, selected_files, freq, plot_time_warped=plot_time_warped, show_peaks=show_peaks)
            st.session_state['current_plots'] = fig_list
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+".pdf" for f in selected_files]
            st.session_state['current_table'] =  [display_threshold_table(selected_dfs, selected_files, [freq]), 
                                                  display_peaks_table(selected_dfs, selected_files, [freq], distinct_dbs)]

        if freqbuttons2.button('Single frequency, stacked', use_container_width=True):
            fig_list = plot_waves_stacked(selected_dfs, selected_files, freq, stacked_labels=stacked_labels)
            st.session_state['current_plots'] = fig_list
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+"_stacked.pdf" for f in selected_files]
            st.session_state['current_table'] =  display_threshold_table(selected_dfs, selected_files, [freq])
        
        if tab2.button("Single dB SPL", use_container_width=True):
            fig_list = plot_waves_single_dB(selected_dfs, selected_files, db, plot_time_warped=plot_time_warped, show_peaks=show_peaks)
            st.session_state['current_plots'] = fig_list
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + str(int(db)) + "dB.pdf" for f in selected_files]
            st.session_state['current_table'] =  None

        if tab2.button("3D surface", use_container_width=True):
            fig_list = plot_3d_surface(selected_dfs, selected_files, freq)
            st.session_state['current_plots'] = fig_list
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_3Dplot.pdf" for f in selected_files]
            st.session_state['current_table'] =  None

        iobuttons1, iobuttons2 = tab2.columns([1, 1.5])
        if iobuttons1.button("I/O curve", use_container_width=True):
            fig_list = plot_io_curve(selected_dfs, selected_files, [freq], distinct_dbs)
            st.session_state['current_plots'] = fig_list
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_IO_plot.pdf" for f in selected_files]
            st.session_state['current_table'] =  None

        if iobuttons2.button("All I/O curves", use_container_width=True):
            fig_list = plot_io_curve(selected_dfs, selected_files, distinct_freqs, distinct_dbs)
            st.session_state['current_plots'] = fig_list
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_IO_plot.pdf" for f in selected_files]
            st.session_state['current_table'] =  None  

        tab2.header("Data outputs:")
        if tab2.button("Return all thresholds", use_container_width=True):
            threshold_table = all_thresholds(selected_dfs, selected_files, distinct_freqs)
            st.session_state['current_plots'] = []
            st.session_state['current_plot_filenames'] = []
            st.session_state['current_table'] =  threshold_table
        
        if tab2.button("Return all peak analyses", use_container_width=True):
            metrics_table = display_metrics_table_all_db(selected_dfs, selected_files, distinct_freqs, distinct_dbs)
            st.session_state['current_plots'] = []
            st.session_state['current_plot_filenames'] = []
            st.session_state['current_table'] =  metrics_table

        #st.markdown(get_download_link(fig), unsafe_allow_html=True)
        if 'current_plots' in st.session_state and st.session_state['current_plots']:
            for i, fig in enumerate(st.session_state['current_plots']): # in range(len(fig_list)):
                st.plotly_chart(fig)
                buffer = io.BytesIO()
                fig.write_image(file=buffer, format="pdf")

                st.download_button(
                    label="Download plot as PDF",
                    data=buffer,
                    file_name=st.session_state['current_plot_filenames'][i],
                    mime="application/pdf",
                    key=f'plot_download_{i}'
                )
        if 'current_table' in st.session_state and st.session_state['current_table'] is not None:
            metrics_table = st.session_state['current_table']
            if isinstance(metrics_table, list):
                for table in metrics_table:
                    st.data_editor(table, hide_index=True, use_container_width=True)
            else:
                st.dataframe(metrics_table, hide_index=True, use_container_width=True)


    else:
        tab2.write("Please upload files to analyze in the 'Data' tab.")

    st.sidebar.caption("[preprint](https://www.biorxiv.org/content/10.1101/2024.06.20.599815v2) | [github](https://github.com/ucsdmanorlab/abranalysis)")

if __name__ == "__main__":
    main()