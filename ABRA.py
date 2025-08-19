import datetime
import io
import os
import struct
import tempfile
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
from utils.preprocessing import interpolate_and_smooth, calculate_and_plot_wave, display_metrics_table_all_db, calculate_hearing_threshold
from utils.plotting import apply_units, style_layout, add_peaks_troughs
from utils.settings import InputSettings, OutputSettings
import warnings
warnings.filterwarnings('ignore')

# TODO: use session states so plots don't disappear when downloading files
# TODO: consider converting freqs to kHz throughout for readability
# TODO: correct units 
# TODO: fix IO curve bug for multiple tsv files
# TODO: switch from dictionary to dataclass for settings

# Co-authored by: Abhijeeth Erra and Jeffrey Chen

def createSettings_DC():
    input_settings = InputSettings(
        time_scale=time_scale,
        level=level,
        units=units,
        calibration_levels=calibration_levels,
        multiply_y_factor=multiply_y_factor
    )
    output_settings = OutputSettings(
        return_units=return_units,
        smooth_on=smooth_on,
        all_peaks=all_peaks,
        serif_font=serif_font,
        show_peaks=show_peaks,
        show_legend=show_legend,
        plot_time_warped=plot_time_warped,
        auto_y=auto_y,
        y_min=y_min,
        y_max=y_max,
        vert_space=vert_space,
    )
    return input_settings, output_settings
def createSettings():
    input_settings = {
            'time_scale': time_scale,
            'level': level,
            'units': units,
            'calibration_levels': calibration_levels,
            'multiply_y_factor': multiply_y_factor
        }
    output_settings = {
            'return_units': return_units,
            'smooth_on': smooth_on,
            'all_peaks': all_peaks,
            'serif_font': serif_font,
            'show_peaks': show_peaks,
            'show_legend': show_legend,
            'plot_time_warped': plot_time_warped,
            'auto_y': auto_y,
            'y_min': y_min,
            'y_max': y_max,
            'vert_space': vert_space,
        }
    return input_settings, output_settings

def plot_waves_single_dB(selected_dfs, selected_files, db, y_min, y_max, input_settings, output_settings, plot_time_warped=False):
    db_column = 'Level(dB)' if input_settings['level'] else 'PostAtten(dB)'
    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return
    
    fig_list = []
    for idx, file_df in enumerate(selected_dfs):
        if db not in file_df[db_column].unique():
            st.write(f"dB {db} not found in file {selected_files[idx].split('/')[-1]}")
            continue
        fig = go.Figure()
        df_filtered = file_df[file_df[db_column] == db]
        freqs = sorted(df_filtered['Freq(Hz)'].unique())

        glasbey_colors = cc.glasbey[:len(freqs)]

        original_waves = []

        for i, freq in enumerate(sorted(freqs)):
            x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, input_settings=input_settings, output_settings=output_settings)

            if y_values is not None:
                y_values = apply_units(y_values, output_settings)
                
                if plot_time_warped:
                    original_waves.append(y_values.tolist())
                    continue

                color = glasbey_colors[i]
                width = 2
                name = freq if type(freq)==str else f'{int(freq)} Hz'
                
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=name, line=dict(color=color, width=width)))
                
                if show_peaks:
                    # Mark the highest peaks with red markers
                    fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks', showlegend=show_legend))

                    # Mark the relevant troughs with blue markers
                    fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs', showlegend=show_legend))

        if plot_time_warped:
            original_waves_array = np.array([wave[:-1] for wave in original_waves])
            try:
                time = np.linspace(0, time_scale, original_waves_array.shape[1])
                obj = fs.fdawarp(original_waves_array.T, time)
                obj.srsf_align(parallel=True)
                warped_waves_array = obj.fn.T
                for i, freq in enumerate(freqs):
                    color = glasbey_colors[i]
                    width = 2
                    name = freq if type(freq)==str else f'{int(freq)} Hz'
                    fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(warped_waves_array[i])), y=warped_waves_array[i], mode='lines', name=name, line=dict(color=color, width=width)))
            except IndexError:
                pass

        fig = style_layout(fig,
                           f'{selected_files[idx].split("/")[-1]} - {db} dB',
                           output_settings)                           
        fig_list.append(fig)
        #except Exception as e:
        #        st.write(f"Error processing freq {freq}: for file {selected_files[idx]} {e}")
    return fig_list

def plot_waves_single_frequency(selected_dfs, selected_files, freq, y_min, y_max, input_settings, output_settings, plot_time_warped=False):
    db_column = 'Level(dB)' if input_settings['level'] else 'PostAtten(dB)'

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return
    
    fig_list = []
    for idx, file_df in enumerate(selected_dfs):
        # check if frequency exists in df:
        if freq not in file_df['Freq(Hz)'].unique():
            st.write(f"Frequency {freq} not found in file {selected_files[idx].split('/')[-1]}")
            continue
        #try:
        fig = go.Figure()
        df_filtered = file_df[file_df['Freq(Hz)'] == freq]
        db_levels = sorted(df_filtered[db_column].unique())
        glasbey_colors = cc.glasbey[:len(db_levels)]

        original_waves = []

        try:
            threshold = np.abs(calculate_hearing_threshold(file_df, freq, input_settings=input_settings))
        except Exception as e:
            threshold = None
         
        for i, db in enumerate(sorted(db_levels)):
            x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, input_settings=input_settings, output_settings=output_settings)
            
            if y_values is not None:
                y_values = apply_units(y_values, output_settings)
                if output_settings['plot_time_warped']:
                    original_waves.append(y_values.tolist())
                    continue

                color = 'black' if db == threshold else glasbey_colors[i]
                width = 5 if db == threshold else 2
                name = f'{int(db)} dB' if db_column == 'Level(dB)' else f'{calibration_levels[(file_df.name, freq)] - int(db)} dB'
                if db == threshold:
                    name = 'Threshold: ' + name
                
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=name, line=dict(color=color, width=width)))

                if output_settings['show_peaks']:
                    # Mark the highest peaks with red markers
                    fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks', showlegend=show_legend))

                    # Mark the relevant troughs with blue markers
                    fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs', showlegend=show_legend))

        if output_settings['plot_time_warped']:
            original_waves_array = np.array([wave[:-1] for wave in original_waves])
            try:
                time = np.linspace(0, time_scale, original_waves_array.shape[1])
                obj = fs.fdawarp(original_waves_array.T, time)
                obj.srsf_align(parallel=True)
                warped_waves_array = obj.fn.T
                for i, db in enumerate(db_levels):
                    color = 'black' if db == threshold else glasbey_colors[i]
                    width = 5 if db == threshold else 2
                    name = f'{int(db)} dB' if db_column == 'Level(dB)' else f'{calibration_levels[(file_df.name, freq)] - int(db)} dB'
                    if db == threshold:
                        name = 'Threshold: ' + name
                    fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(warped_waves_array[i])), y=warped_waves_array[i], mode='lines', name=name, line=dict(color=color, width=width)))
            except IndexError:
                pass
        
        fig = style_layout(fig,
                           f'{selected_files[idx].split("/")[-1]} - Frequency: {freq} Hz',
                           output_settings)
        fig_list.append(fig)
        #except Exception as e:
        #        st.write(f"Error processing freq {freq}: for file {selected_files[idx]} {e}")
    return fig_list

def plot_waves_single_tuple(selected_dfs, selected_files, freq, db, y_min, y_max, input_settings, output_settings):
    fig = go.Figure()

    for idx, file_df in enumerate(selected_dfs):
        x_values, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db, input_settings=input_settings, output_settings=output_settings)
        if y_values is not None:
            y_values = apply_units(y_values, output_settings)

            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'{selected_files[idx].split("/")[-1]}'))#, showlegend=False))
            if output_settings['show_peaks']:
                # Mark the highest peaks with red markers
                fig.add_trace(go.Scatter(x=x_values[highest_peaks], y=y_values[highest_peaks], mode='markers', marker=dict(color='red'), name='Peaks'))#, showlegend=False))

                # Mark the relevant troughs with blue markers
                fig.add_trace(go.Scatter(x=x_values[relevant_troughs], y=y_values[relevant_troughs], mode='markers', marker=dict(color='blue'), name='Troughs'))#, showlegend=False))

    figtitle = f'{selected_files[idx].split("/")[-1]}, Freq = {freq}, db = {db}' if input_settings['level'] else f'{selected_files[idx].split("/")[-1]}, Freq = {freq}, db = {input_settings['calibration_levels'][(file_df.name, freq)] - int(db)}'
    fig = style_layout(fig,
                       figtitle,
                       output_settings)
    return fig

def plot_3d_surface(selected_dfs, selected_files, freq, input_settings=None, output_settings=None):
    db_column = 'Level(dB)' if input_settings['level'] else 'PostAtten(dB)'

    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    fig_list = []
    for idx, file_df in enumerate(selected_dfs):
        fig = go.Figure()
        df_filtered = file_df[file_df['Freq(Hz)'] == freq]
        if db_column == 'Level(dB)':
            db_levels = sorted(df_filtered[db_column].unique(), reverse=True)
        else:
            db_levels = sorted([calibration_levels[(file_df.name, freq)] - db for db in df_filtered[db_column].unique()], reverse=True)
        
        original_waves = []

        try:
            threshold = calculate_hearing_threshold(file_df, freq, input_settings=input_settings)
        except:
            threshold = None

        for db in db_levels:
            if db_column == 'Level(dB)':
                x_values, y_values, _, _ = calculate_and_plot_wave(file_df, freq, db, input_settings=input_settings, output_settings=output_settings)
            else:
                x_values, y_values, _, _ = calculate_and_plot_wave(file_df, freq, calibration_levels[(file_df.name, freq)] - db, input_settings=input_settings, output_settings=output_settings)

            if y_values is not None:
                y_values = apply_units(y_values, output_settings)
                original_waves.append(y_values.tolist())

        original_waves_array = np.array([wave[:-1] for wave in original_waves])

        try:
            time = np.linspace(0, time_scale, original_waves_array.shape[1])
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
        fig.update_layout(font_family="Times New Roman" if output_settings['serif_font'] else "sans-serif",
                      font_color="black",
                      title_font_family="Times New Roman" if output_settings['serif_font'] else "sans-serif",
                      font=dict(size=14))
        fig.update_layout(showlegend=show_legend)
        fig_list.append(fig)
    return fig_list

def plot_waves_stacked(selected_dfs, selected_files, freq, stacked_labels=None, input_settings=None, output_settings=None):
    if len(selected_dfs) == 0:
        st.write("No files selected.")
        return

    db_column = 'Level(dB)' if level else 'PostAtten(dB)'

    fig_list = []
    for idx, file_df in enumerate(selected_dfs):
        fig = go.Figure()
        if freq not in file_df['Freq(Hz)'].unique():
            st.write(f"Frequency {freq} not found in file {selected_files[idx].split('/')[-1]}")
            continue
        # Get unique dB levels and color palette
        df_filtered = file_df[file_df['Freq(Hz)'] == freq]
        unique_dbs = sorted(df_filtered[db_column].unique())
        if not level:
            unique_dbs = sorted(unique_dbs, reverse=True)
        num_dbs = len(unique_dbs)
        vertical_spacing = output_settings['vert_space'] / num_dbs
        # Get y_min from input_settings if available, else default to 0.0
        y_min = input_settings['y_min'] if input_settings and 'y_min' in input_settings else 0.0
        db_offsets = {db: y_min + i * vertical_spacing for i, db in enumerate(unique_dbs)}
        glasbey_colors = cc.glasbey[:num_dbs]

        # Calculate the hearing threshold
        try:
            threshold = calculate_hearing_threshold(file_df, freq, input_settings=input_settings)
        except:
            threshold = None

        db_levels = sorted(unique_dbs, reverse=True) if db_column == 'Level(dB)' else sorted(unique_dbs)
        if db_column == 'PostAtten(dB)':
            db_levels = np.array(db_levels)
            calibration_level = np.full(len(db_levels), input_settings['calibration_levels'][(df.name, freq)])
            db_levels = calibration_level - db_levels
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
                    final *= input_settings['multiply_y_factor']

                    # if units == 'Nanovolts':
                    #     final /= 1000

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

                    if (db_column == 'Level(dB)' and db == threshold) or (db_column == 'PostAtten(dB)' and calibration_level[0] - db == threshold):
                        is_thresh=True
                        fig.add_trace(go.Scatter(x=np.linspace(0, time_scale, len(y_values)),
                                                y=y_values,
                                                mode='lines',
                                                name=f'Thresh: {int(db)} dB',
                                                line=dict(color='black', width=5),
                                                #showlegend=True
                                                ))
                        
                    y_pos = db_offsets[db] if input_settings['level'] else db_offsets[calibration_level[0] - db]
                    if stacked_labels=="Left outside":
                        x_pos = 0; 
                    elif stacked_labels=="Right outside":
                        x_pos = time_scale*1.1; 
                    elif stacked_labels=="Right inside":
                        x_pos = input_settings['time_scale']; y_pos = y_pos + vert_space/num_dbs/3 #y_values.iloc[-1]
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
                           output_settings,
        )
        fig.update_layout(width=400, height=700,
                          yaxis_title='Voltage (μV)',
                          yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                          xaxis=dict(showgrid=True, zeroline=False))

        khz = file_df[(file_df['Freq(Hz)'] == freq)]
        if not khz.empty:
            fig_list.append(fig)
    return fig_list

def CFTSread(PATH):

    with open(PATH, 'r', encoding='latin1') as file:
        data = file.readlines()

    data_start = False
    data_list = []
    for line in data: 
        if not data_start and line.startswith(':'): # Header lines
            if 'FREQ' in line:
                try:
                    freqs = [float(line.split('FREQ:')[1].split()[0].strip())*1000]
                except: #Click
                    freqs = [line.split('FREQ:')[1].split()[0].strip()]
            if 'LEVELS' in line:
                dbs = (line.split('LEVELS:')[1].strip()[:-1])
                dbs = [int(dB) for dB in dbs.split(';')]
            if 'SAMPLE' in line:
                sample_us = float(line.split('SAMPLE')[1].split(':')[1].strip())
            if 'DATA' in line:
                data_start = True

        elif data_start:
            if len(line.strip()) == 0:
                continue
            data_list.append([float(d) for d in line.strip().split()])

    data = np.array(data_list)

    duration_ms = (data.shape[0]-1) * sample_us/1000
    rows = []
    
    for dB_i in range(len(dbs)):
        db = dbs[dB_i]
        data_col = data[:, dB_i]
        
        wave_data = {f'{i}': data_col[i] for i in range(len(data_col))}#, v in data_col} 
        row = {'Freq(Hz)': freqs[0], 'Level(dB)': db, **wave_data}
        rows.append(row)
        
        df = pd.DataFrame(rows)
    return duration_ms, df

def arfread(PATH, **kwargs):
    def get_str(data):
        # return string up until null character only
        ind = data.find(b'\x00')
        if ind > 0:
            data = data[:ind]
        return data.decode('utf-8')
    # defaults
    RP = kwargs.get('RP', False)
    
    isRZ = not RP
    
    data = {'metadata': {}, 'groups': []}

    # open file
    with open(PATH, 'rb') as fid:
        # open metadata data
        data['metadata']['ftype'] = struct.unpack('h', fid.read(2))[0]
        data['metadata']['ngrps'] = struct.unpack('h', fid.read(2))[0]
        data['metadata']['nrecs'] = struct.unpack('h', fid.read(2))[0]
        data['metadata']['grpseek'] = struct.unpack('200i', fid.read(4*200))
        data['metadata']['recseek'] = struct.unpack('2000i', fid.read(4*2000))
        data['metadata']['file_ptr'] = struct.unpack('i', fid.read(4))[0]

        data['groups'] = []
        bFirstPass = True
        for x in range(data['metadata']['ngrps']):
            # jump to the group location in the file
            fid.seek(data['metadata']['grpseek'][x], 0)

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

    return data




def all_thresholds(selected_dfs, selected_files, distinct_freqs, input_settings=None):
    df_dict = {'Filename': [],
               'Frequency': [],
               'Threshold': []}
    for (file_df, file_name) in zip(selected_dfs, selected_files):
        for hz in distinct_freqs:
            if hz not in file_df['Freq(Hz)'].unique():
                continue
            # df_filtered = file_df[(file_df['Freq(Hz)'] == hz)]
            # if df_filtered.empty:
            #     continue
            try:
                thresh = calculate_hearing_threshold(file_df, hz, input_settings=input_settings)
            except:
                thresh = np.nan
                pass
            df_dict['Filename'].append(file_name.split("/")[-1])
            df_dict['Frequency'].append(hz)
            df_dict['Threshold'].append(thresh)
    threshold_table = pd.DataFrame(df_dict)
    st.dataframe(threshold_table, hide_index=True, use_container_width=True)
    return threshold_table



def calculate_unsupervised_threshold(df, freq): # UNUSED
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

            final = final * multiply_y_factor
            if units == 'Nanovolts':
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

def plot_io_curve(selected_dfs, selected_files, freqs, db_levels, input_settings=None, output_settings=None):
    db_column = 'Level(dB)' if input_settings['level'] else 'PostAtten(dB)'

    amplitudes = {}

    ru = 'μV'
    if output_settings['return_units'] == 'Nanovolts':
        ru = 'nV'

    fig_list = []
    for file_df, file_name in zip(selected_dfs, selected_files):
        for freq in freqs:
            for db in db_levels:
                _, y_values, highest_peaks, relevant_troughs = calculate_and_plot_wave(file_df, freq, db)

                if highest_peaks is not None:
                    if highest_peaks.size > 0:  # Check if highest_peaks is not empty
                        y_values = apply_units(y_values, output_settings)
                        first_peak_amplitude = y_values[highest_peaks[0]] - y_values[relevant_troughs[0]]
                        if level:
                            amplitudes[db] = first_peak_amplitude
                        else:
                            amplitudes[calibration_levels[(file_df.name, freq)] - int(db)] = first_peak_amplitude

            # Plotting
            fig = go.Figure()
            if level:
                fig.add_trace(go.Scatter(x=sorted(list(amplitudes.keys())), y=list(amplitudes.values()), mode='lines+markers', name=f'Freq: {freq} Hz'))
            else:
                fig.add_trace(go.Scatter(x=np.full(len(db_levels), calibration_levels[(file_df.name, freq)]) - db_levels, y=amplitudes, mode='lines+markers', name=f'Freq: {freq} Hz'))
            
            file_name = file_name.split('/')[-1]
            fig.update_layout(
                title=f'{file_name} I/O Curve for Frequency {freq} Hz',
                xaxis_title='dB Level',
                yaxis_title=f'Wave 1 Amplitude ({ru})',
                xaxis=dict(tickmode='linear', dtick=5),
                yaxis=dict(range=[0, max(amplitudes.values()) + 0.1 * abs(max(amplitudes.values()))]),
                template='plotly_white'
            )
            fig.update_layout(font_family="Times New Roman" if serif_font else "sans-serif",
                            font_color="black",
                            title_font_family="Times New Roman" if serif_font else "sans-serif",
                            font=dict(size=24))

            fig_list.append(fig)
    return fig_list

def main():
    global time_scale, units, is_atten, click, dfs, selected_files, selected_dfs, calibration_levels
    global level, db_column, distinct_freqs, distinct_dbs, multiply_y_factor, return_units, show_legend, show_peaks
    global smooth_on, all_peaks, serif_font, vert_space, stacked_labels, plot_time_warped, auto_y, y_min, y_max
    
    # Streamlit UI
    st.title("ABRA")
    tab1, tab2 = st.sidebar.tabs(["Data", "Plotting and Analysis"])
    
    uploaded_files = tab1.file_uploader("**Upload files to analyze:**", type=["csv", "arf", "asc", "tsv"], accept_multiple_files=True)
    is_rz_file = "RZ"
    
    # Inputs:
    inputs = tab1.expander("Input data properties", expanded=True)
    placeholder = inputs.empty()
    units = inputs.selectbox("Units used in collection", options=['Microvolts', 'Nanovolts'], index=0)
    # baseline_level_str = inputs.text_input("Set Baseline Level", "0.0")
    # baseline_level = float(baseline_level_str)
    is_click = inputs.radio("Tone or click? (for .arf files)", ("Tone", "Click"), horizontal=True)
    click = True if is_click == "Click" else False
    is_atten = inputs.toggle("dB saved as attenuation", value=False)

    if uploaded_files:
        dfs = []
        selected_files = []
        selected_dfs = []
        calibration_levels = {}

        duration = None
        # st.sidebar.write("Select files to analyze:")
        for idx, file in enumerate(uploaded_files):
            # Use tempfile
            temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file.read())
            #st.sidebar.markdown(f"**File Name:** {file.name}")
            selected = tab2.checkbox(f"{file.name}", key=f"file_{idx}", value=True)
            
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
                        
                        if not is_atten:
                            row = {'Freq(Hz)': freq, 'Level(dB)': db, **wave_data}
                            rows.append(row)
                        else:
                            row = {'Freq(Hz)': freq, 'PostAtten(dB)': db, **wave_data}
                            rows.append(row)
                        duration = rec['dur_ms']

                df = pd.DataFrame(rows)
            elif file.name.endswith(".asc") or file.name.endswith(".tsv"):
                # Process ASC file
                duration, df = CFTSread(temp_file_path)

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

        if duration is not None:
            time_scale = placeholder.number_input("Time scale of recording (detected from file, ms)", value=duration, format="%0.6f")
        else:
            time_scale = placeholder.number_input("Time scale of recording (ms)", value=10.0)

        level = (not is_atten)

        db_column = 'Level(dB)' if level else 'PostAtten(dB)'

        # Get distinct frequency and dB level values across all files
        distinct_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in dfs]).unique())
        distinct_dbs = sorted(pd.concat([df['Level(dB)'] if level else df['PostAtten(dB)'] for df in dfs]).unique())

        if not level:
            cal_levels = tab1.expander("Calibration dB levels", expanded=True)
            # TODO: ask if this needs to be set per file or should be generalized across all files??
            for file in selected_files: 
                for hz in distinct_freqs:
                    key = (os.path.basename(file), hz)
                    calibration_levels[key] = cal_levels.number_input(f"Calibration dB for {os.path.basename(file)} at {hz} Hz", 
                                                                    value=100.0, step=5.0, format="%0.1f",)
            # TODO: add option to set legend to show attenuation
            #atten_legend = cal_levels.toggle("Use attenuation levels in plot legends", value=False)

        # Output settings:
        outputs = tab2.expander("Output and plot settings", expanded=False)
        return_units = outputs.selectbox("Units for plots and outputs", options=['Microvolts', 'Nanovolts'], index=0)
        if return_units == 'Nanovolts':
            ymin = -5000.0
            ymax = 5000.0
        else:
            ymin = -5.0
            ymax = 5.0
        auto_y = outputs.toggle("Auto Y-axis scaling", value=True)
        y_min = outputs.number_input("Y-axis minimum", value=ymin, disabled=auto_y)
        y_max = outputs.number_input("Y-axis maximum", value=ymax, disabled=auto_y)
        plot_time_warped = outputs.toggle("Plot time warped curves", False)
        show_legend = outputs.toggle("Show legend", True)
        show_peaks = outputs.toggle("Show peaks (single wave and single frequency plots)", True)
        serif_font = outputs.toggle("Use serif fonts in plots", value=False)

        advanced_settings = tab2.expander("Advanced settings", expanded=False)
        multiply_y_factor = advanced_settings.number_input("Multiply Y values by factor", value=1.0)
        vert_space = advanced_settings.number_input("Vertical space (for stacked curves)", value=10.0, min_value=0.0, step=1.0)
        stacked_labels = advanced_settings.selectbox("Stacked labels position", options=["Left outside", "Right outside", "Right inside", "Off"], index=2)
        all_peaks = advanced_settings.toggle("Output all peaks and troughs (experimental)", value=False)
        smooth_on = advanced_settings.toggle("Smooth wave", value=True)
        
        # Frequency dropdown options
        allowed_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in selected_dfs]).unique())
        freq = tab2.selectbox("Select frequency (Hz)", options=allowed_freqs, index=0)
        # dB Level dropdown options, default to last (highest) dB)
        allowed_dbs = sorted(pd.concat([df[df['Freq(Hz)']==freq]['Level(dB)'] if level else df[df['Freq(Hz)']==freq]['PostAtten(dB)'] for df in selected_dfs]).unique())
        db = tab2.selectbox("Select sound level (dB)" if level else "Select sound level (attenuation, dB)", 
                            options=allowed_dbs, 
                            index=len(allowed_dbs)-1 if len(allowed_dbs) > 0 and level else 0)
        
        # Create a plotly figure
        #fig = go.Figure()
        tab2.header("Plot:")
        freq_str = freq if type(freq) == str else str(freq/1000) + " kHz"
        db_str = str(int(db)) + " dB"
        if tab2.button("Single wave ("+freq_str+", "+db_str+")", use_container_width=True):
            input_settings, output_settings = createSettings()
            fig = plot_waves_single_tuple(selected_dfs, selected_files,freq, db, y_min, y_max, input_settings=input_settings, output_settings=output_settings)
            st.plotly_chart(fig)
            
            metrics_table = display_metrics_table_all_db(selected_dfs, selected_files, [freq], [db], input_settings=input_settings, output_settings=output_settings)
            
            st.dataframe(metrics_table, hide_index=True, use_container_width=True)
            # Create an in-memory buffer
            buffer = io.BytesIO()

            # Save the figure as a pdf to the buffer
            fig.write_image(file=buffer, format="pdf")

            # Download the pdf from the buffer
            st.download_button(
                label="Download plot as PDF",
                data=buffer,
                file_name=(selected_files[0].split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_"+db_str.replace(' ','')+".pdf") if len(selected_files)==1 else "all_files_" +freq_str.replace(' ','')+ "_"+db_str.replace(' ','')+".pdf",
                mime="application/pdf",
            )
        
        freqbuttons1, freqbuttons2 = tab2.columns([1, 1.5])
        if freqbuttons1.button("Single frequency", use_container_width=True):
            input_settings, output_settings = createSettings()

            fig_list = plot_waves_single_frequency(selected_dfs, selected_files, freq, y_min, y_max, input_settings=input_settings, output_settings=output_settings, plot_time_warped=plot_time_warped)

            for i in range(len(fig_list)):
                st.plotly_chart(fig_list[i])
            
                buffer = io.BytesIO()

                # Save the figure as a pdf to the buffer
                fig_list[i].write_image(file=buffer, format="pdf")

                # Download the pdf from the buffer
                st.download_button(
                    label="Download plot as PDF",
                    data=buffer,
                    file_name=selected_files[i].split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+".pdf",
                    mime="application/pdf",
                    key=f'file{i}'
                )
            metrics_table = display_metrics_table_all_db(selected_dfs, selected_files, [freq], distinct_dbs, input_settings=input_settings, output_settings=output_settings)

            st.dataframe(metrics_table, hide_index=True, use_container_width=True)

        
        if freqbuttons2.button('Single frequency, stacked', use_container_width=True):
            input_settings, output_settings = createSettings()

            fig_list = plot_waves_stacked(selected_dfs, selected_files, freq, stacked_labels=stacked_labels, input_settings=input_settings, output_settings=output_settings)
            for i in range(len(fig_list)):
                st.plotly_chart(fig_list[i])
            
                buffer = io.BytesIO()

                # Save the figure as a pdf to the buffer
                fig_list[i].write_image(file=buffer, format="pdf")
                
                # Download the pdf from the buffer
                st.download_button(
                    label="Download plot as PDF",
                    data=buffer,
                    file_name=selected_files[i].split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_stacked.pdf",
                    mime="application/pdf",
                    key=f'file{i}'
                )
        
        if tab2.button("Single dB", use_container_width=True):
            input_settings, output_settings = createSettings()

            fig_list = plot_waves_single_dB(selected_dfs, selected_files, db, y_min, y_max, input_settings=input_settings, output_settings=output_settings, plot_time_warped=plot_time_warped)

            for i in range(len(fig_list)):
                st.plotly_chart(fig_list[i])
            
                buffer = io.BytesIO()

                # Save the figure as a pdf to the buffer
                fig_list[i].write_image(file=buffer, format="pdf")

                # Download the pdf from the buffer
                st.download_button(
                    label="Download plot as PDF",
                    data=buffer,
                    file_name=selected_files[i].split("/")[-1].split('.')[0] + "_" + str(int(db))+"dB.pdf",
                    mime="application/pdf",
                    key=f'file{i}'
                )
            # display_metrics_table_all_db(selected_dfs, [freq], distinct_dbs, time_scale)
            # need to add table functionality

        if tab2.button("3D surface", use_container_width=True):
            input_settings, output_settings = createSettings()

            fig_list = plot_3d_surface(selected_dfs, selected_files, freq, input_settings=input_settings, output_settings=output_settings)
            for i in range(len(fig_list)):
                st.plotly_chart(fig_list[i])
            
                buffer = io.BytesIO()

                # Save the figure as a pdf to the buffer
                fig_list[i].write_image(file=buffer, format="pdf")

                # Download the pdf from the buffer
                st.download_button(
                    label="Download plot as PDF",
                    data=buffer,
                    file_name=selected_files[i].split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_3Dplot.pdf",
                    mime="application/pdf",
                    key=f'file{i}'
                )

        #io_all_freqs = st.sidebar.toggle("All frequencies", value=False)
        if tab2.button("I/O curve", use_container_width=True):
            # if io_all_freqs:
            #     fig_list = plot_io_curve(df, distinct_freqs, distinct_dbs)
            # else:
            input_settings, output_settings = createSettings()

            fig_list = plot_io_curve(selected_dfs, selected_files, [freq], distinct_dbs, input_settings=input_settings, output_settings=output_settings)
            
            for i in range(len(fig_list)):
                st.plotly_chart(fig_list[i])
            
                buffer = io.BytesIO()

                # Save the figure as a pdf to the buffer
                fig_list[i].write_image(file=buffer, format="pdf")

                # Download the pdf from the buffer
                st.download_button(
                    label="Download plot as PDF",
                    data=buffer,
                    file_name=selected_files[i].split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+"IO_plot.pdf",
                    mime="application/pdf",
                    key=f'file{i}'
                )

        tab2.header("Data outputs:")
        if tab2.button("Return all thresholds", use_container_width=True):
            input_settings, _ = createSettings()
            
            all_thresholds(selected_dfs, selected_files, distinct_freqs, input_settings=input_settings)
        
        if tab2.button("Return all peak analyses", use_container_width=True):
            input_settings, output_settings = createSettings()
            metrics_table = display_metrics_table_all_db(selected_dfs, selected_files, distinct_freqs, distinct_dbs, input_settings=input_settings, output_settings=output_settings)
            st.dataframe(metrics_table, hide_index=True, use_container_width=True)

        #st.markdown(get_download_link(fig), unsafe_allow_html=True)

    else:
        tab2.write("Please upload files to analyze in the 'Data' tab.")

    st.sidebar.caption("[preprint](https://www.biorxiv.org/content/10.1101/2024.06.20.599815v2) | [github](https://github.com/ucsdmanorlab/abranalysis)")

if __name__ == "__main__":
    main()