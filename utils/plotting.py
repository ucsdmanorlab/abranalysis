import streamlit as st
from .calculate import *
import plotly.graph_objects as go
import numpy as np
import fdasrsf as fs
import colorcet as cc
import pandas as pd

def db_column_name():
    atten = st.session_state.get('atten', False)
    return 'Level(dB)' if not atten else 'PostAtten(dB)'

def db_value(file_name, freq, db):
    atten = st.session_state.get('atten', False)
    if atten:
        return st.session_state.calibration_levels[(file_name, freq)] - int(db)
    else:
        return db
    
def apply_units(y_values):
    return y_values * 1000 if st.session_state.return_units == 'Nanovolts' else y_values

def get_y_units():
    return 'Voltage (nV)' if st.session_state.return_units == 'Nanovolts' else 'Voltage (μV)'

def style_layout(fig, title):
    fig.update_layout(
        title=title,
        xaxis_title='Time (ms)',
        yaxis_title=get_y_units(),
        width=700, height=450,
        font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
        font_color="black",
        title_font_family="Times New Roman" if st.session_state.serif_font else "sans-serif",
        font=dict(size=18),
        showlegend=st.session_state.show_legend
    )
    if not st.session_state.auto_y:
        fig.update_layout(yaxis_range=[st.session_state.y_min, st.session_state.y_max])
    return fig

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

def plot_3d_surface(selected_dfs, selected_files, freq, plot_time_warped=False):
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
        time = np.linspace(0, st.session_state.time_scale, original_waves_array.shape[1])

        if plot_time_warped:    
            try:
                time = np.linspace(0, st.session_state.time_scale, original_waves_array.shape[1])
                obj = fs.fdawarp(original_waves_array.T, time)
                obj.srsf_align(parallel=True)
                warped_waves_array = obj.fn.T
            except IndexError:
                warped_waves_array = np.array([])
        else:
            warped_waves_array = original_waves_array

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
