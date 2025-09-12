import datetime
import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from utils.calculate import *
from utils.plotting import *
from utils.processFiles import get_selected_data, process_uploaded_files_cached, db_column_name

import warnings
warnings.filterwarnings('ignore')

# TODO: consider converting freqs to kHz throughout for readability
# TODO: correct units
# TODO: make 3D plots work for tsv files

# Co-authored by: Abhijeeth Erra and Jeffrey Chen
def clear_plots_and_tables():
    st.session_state['current_plots'] = []
    st.session_state['current_plot_filenames'] = []
    st.session_state['current_table'] = None
    st.session_state['threshold_table'] = None
    st.session_state['peaks_table'] = None
    st.session_state['peak_editor_table'] = None
    st.session_state.editing_peaks_table = False

def manual_threshold_warning():
    if 'manual_thresholds' in st.session_state and st.session_state.manual_thresholds:
        col1, col2 = st.columns([2, 1])
        col1.warning("Manual thresholds are set.")
        if col2.button("Clear manual thresholds"):
            st.session_state.manual_thresholds.clear()
            st.success("Manual thresholds cleared. Choose plot option to refresh.")

def check_settings_and_clear_cache():
    calc_settings = {
        'time_scale': st.session_state.get('time_scale', 10.0),
        'multiply_y_factor': st.session_state.get('multiply_y_factor', 1.0),
        'units': st.session_state.get('units', 'Microvolts'),
        'smooth_on': st.session_state.get('smooth_on', True),
        'return_units': st.session_state.get('return_units', 'Microvolts'),
        'calibration_levels': dict(st.session_state.get('calibration_levels', {}),),
        'peaks_below_thresh': st.session_state.get('peaks_below_thresh', False),
    }
    
    # Check if settings changed
    if 'previous_calc_settings' in st.session_state:
        if st.session_state.previous_calc_settings != calc_settings:
            st.warning("Settings have changed. Cleared calculated values.")
            if 'calculated_thresholds' in st.session_state:
                st.session_state.calculated_thresholds.clear()
            if 'calculated_waves' in st.session_state:
                st.session_state.calculated_waves.clear()
            if 'manual_thresholds' in st.session_state:
                st.session_state.manual_thresholds.clear()
            if 'manual_peaks' in st.session_state:
                st.session_state.manual_peaks.clear()
            clear_plots_and_tables()
    
    st.session_state.previous_calc_settings = calc_settings
    

def main():    
    if 'calibration_levels' not in st.session_state:
        st.session_state.calibration_levels = {}

    # Streamlit UI
    st.title("ABRA")
    tab1, tab2 = st.sidebar.tabs(["Data", "Plotting and Analysis"])
    
    uploaded_files = tab1.file_uploader("**Upload files to analyze:**", type=["csv", "arf", "asc", "tsv"], accept_multiple_files=True)
    
    # Inputs:
    inputs = tab1.expander("Input data properties", expanded=True)
    placeholder = inputs.empty()
    units = inputs.selectbox("Units used in collection", options=['Microvolts', 'Nanovolts'], index=0, key="units")
    # baseline_level_str = inputs.text_input("Set Baseline Level", "0.0")
    # baseline_level = float(baseline_level_str)
    is_click = inputs.radio("Tone or click? (for .arf files)", ("Tone", "Click"), horizontal=True)
    click = True if is_click == "Click" else False
    # is_rz_file = inputs.radio("BioSig version (.arf files):", ("RZ", "RP"), horizontal=True, index=0) 
    is_atten = inputs.toggle("dB saved as attenuation (.arf only)", value=False, key="atten")

    if uploaded_files:

        dfs, duration = process_uploaded_files_cached(uploaded_files, "RZ", click, is_atten)

        if duration is not None:
            time_scale = placeholder.number_input("Time scale of recording (detected from file, ms)", value=duration, format="%0.6f", key="time_scale")
        else:
            time_scale = placeholder.number_input("Time scale of recording (ms)", value=10.0, key="time_scale")

        tab2.write("**Select files to analyze:**")
        for idx, df in enumerate(dfs):
            tab2.checkbox(f"{df.name}", key=f"file_{idx}", value=True)
            
        selected_files, selected_dfs = get_selected_data()

        level = (not is_atten)

        if dfs:
            distinct_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in dfs]).unique())
            distinct_dbs = sorted(pd.concat([df[db_column_name()] for df in dfs]).unique())
        else:
            distinct_freqs, distinct_dbs = [], []

        if is_atten:
            cal_levels = tab1.expander("Calibration dB levels", expanded=True)
            for file_path in selected_files: 
                file_name = os.path.basename(file_path)
                for hz in distinct_freqs:
                    key = hz
                    if key not in st.session_state.calibration_levels:
                        st.session_state.calibration_levels[key] = 100.0
                    st.session_state.calibration_levels[key] = cal_levels.number_input(
                        f"Calibration dB for {hz} Hz",
                        value=st.session_state.calibration_levels[key],
                        step=5.0,
                        format="%0.1f",
                        key=f"cal_{hz}"
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
        peaks_below_thresh = advanced_settings.toggle("Calculate peaks/troughs below threshold", value=False, key="peaks_below_thresh")
        #unfilter_pk1 = advanced_settings.toggle("Include sub-threshold peak 1 amplitudes", value=False, key="unfilter_pk1")

        check_settings_and_clear_cache()

        # Frequency dropdown options
        allowed_freqs = sorted(pd.concat([df['Freq(Hz)'] for df in selected_dfs]).unique()) if selected_dfs else []
        freq = tab2.selectbox("Select frequency (Hz)", options=allowed_freqs, index=0)
        # dB Level dropdown options, default to last (highest) dB)
        allowed_dbs = sorted(pd.concat([df[df['Freq(Hz)']==freq][db_column_name()] for df in selected_dfs]).unique()) if selected_dfs else []
        db = tab2.selectbox("Select sound level (dB)" if level else "Select sound level (attenuation, dB)", 
                            options=allowed_dbs, 
                            index=len(allowed_dbs)-1 if len(allowed_dbs) > 0 and level else 0)
        
        
        freq_str = freq if type(freq) == str else str(freq/1000) + " kHz"
        db_str = str(int(db)) + " dB"
        tab2.header("Plot:")
        if tab2.button("Single wave ("+freq_str+", "+db_str+")", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_plots'] = [plot_waves_single_tuple(selected_dfs, selected_files,freq, db, show_peaks=show_peaks)]
            st.session_state['current_plot_filenames'] = [(selected_files[0].split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_"+db_str.replace(' ','')+".pdf") if len(selected_files)==1 else "all_files_" +freq_str.replace(' ','')+ "_"+db_str.replace(' ','')+".pdf"]
            st.session_state['peaks_table'] =  display_peaks_table(selected_dfs, selected_files, [freq], [db], return_threshold=True, return_nas=True)
            st.session_state['peak_editor_table'] =  display_peaks_table(selected_dfs, selected_files, [freq], [db], return_threshold=True, return_nas=True, editable=True)
            
        freqbuttons1, freqbuttons2 = tab2.columns([1, 1.5])
        if freqbuttons1.button("Single frequency", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_plots'] = plot_waves_single_frequency(selected_dfs, selected_files, freq, plot_time_warped=plot_time_warped, show_peaks=show_peaks)
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+".pdf" for f in selected_files]
            st.session_state['threshold_table'] = display_threshold_table(selected_dfs, selected_files, [freq])
            st.session_state['peaks_table'] = display_peaks_table(selected_dfs, selected_files, [freq], distinct_dbs)
            st.session_state['peak_editor_table'] =  display_peaks_table(selected_dfs, selected_files, [freq], distinct_dbs, return_threshold=True, return_nas=True, editable=True)

        if freqbuttons2.button('Single frequency, stacked', use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_plots'] = plot_waves_stacked(selected_dfs, selected_files, freq, stacked_labels=stacked_labels)
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+"_stacked.pdf" for f in selected_files]
            st.session_state['threshold_table'] =  display_threshold_table(selected_dfs, selected_files, [freq])
            
        if tab2.button("Single dB SPL", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_plots'] = plot_waves_single_dB(selected_dfs, selected_files, db, plot_time_warped=plot_time_warped, show_peaks=show_peaks)
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + str(int(db)) + "dB.pdf" for f in selected_files]
            
        if tab2.button("3D surface", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_plots'] = plot_3d_surface(selected_dfs, selected_files, freq, plot_time_warped=plot_time_warped)
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_3Dplot.pdf" for f in selected_files]
           
        iobuttons1, iobuttons2 = tab2.columns([1, 1.5])
        if iobuttons1.button("I/O curve", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_plots'] = plot_io_curve(selected_dfs, selected_files, [freq], distinct_dbs)
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_" + freq_str.replace(' ','')+ "_IO_plot.pdf" for f in selected_files]
            
        if iobuttons2.button("All I/O curves", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_plots'] = plot_io_curve(selected_dfs, selected_files, distinct_freqs, distinct_dbs)
            st.session_state['current_plot_filenames'] = [f.split("/")[-1].split('.')[0] + "_IO_plot.pdf" for f in selected_files]

        tab2.header("Data outputs:")
        if tab2.button("Return all thresholds", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['threshold_table'] = display_threshold_table(selected_dfs, selected_files, distinct_freqs)
            
        if tab2.button("Return all peak analyses", use_container_width=True):
            clear_plots_and_tables()
            st.session_state['current_table'] =  display_peaks_table(selected_dfs, selected_files, distinct_freqs, distinct_dbs, return_threshold=True)
            # display_metrics_table_all_db(selected_dfs, selected_files, distinct_freqs, distinct_dbs)

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
        lock_cols = ["File Name", "Frequency (Hz)", 'Sound amplitude (dB SPL)', 'Attenuation (dB)', 'Calibration Level (dB)']
        if 'current_table' in st.session_state and st.session_state['current_table'] is not None:
            metrics_table = st.session_state['current_table']
            st.dataframe(metrics_table, hide_index=True, use_container_width=True)
            manual_threshold_warning()
            
        if 'threshold_table' in st.session_state and st.session_state['threshold_table'] is not None:
            threshold_table = st.session_state['threshold_table']
            disable_columns = [col for col in threshold_table.columns if col in lock_cols]
            edited_df = st.data_editor(threshold_table, hide_index=True, use_container_width=True, disabled=disable_columns)
            if (edited_df != threshold_table).any().any():
                if 'manual_thresholds' not in st.session_state:
                    st.session_state.manual_thresholds = {}
                diffs = edited_df.ne(threshold_table)
                edited_positions = np.where(diffs)
                for row_idx in edited_positions[0]:
                    file_name = edited_df.iloc[row_idx]['File Name']
                    freq = edited_df.iloc[row_idx]['Frequency (Hz)']
                    threshold = edited_df.iloc[row_idx]['Estimated Threshold']
                    edit_key = f"{file_name}_{freq}"
                    st.session_state.manual_thresholds[edit_key] = threshold
            manual_threshold_warning()
        if 'peaks_table' in st.session_state and st.session_state['peaks_table'] is not None:
            peaks_table = st.session_state['peaks_table']
            if not st.session_state.editing_peaks_table:
                st.dataframe(peaks_table, hide_index=True, use_container_width=True)
                if 'peak_editor_table' in st.session_state and st.session_state.peak_editor_table is not None:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    if col1.button("Edit peaks table"):
                        st.session_state.editing_peaks_table = True
                        st.rerun()
                    if 'manual_peaks' in st.session_state and st.session_state.manual_peaks:
                        col2.warning("Manual peak edits are set.")
                        if col3.button("Clear manual peak edits"):
                            st.session_state.manual_peaks.clear()
                            st.success("Manual peak edits cleared. Choose plot option to refresh.")
            else:
                col1, col2 = st.columns([1, 1])
                col1.warning("**Editing peaks mode**")

                peak_editor_table = st.session_state['peak_editor_table'] 
                disable_columns = [col for col in peak_editor_table.columns if 'latency' not in col]
                edited_peaks_df = st.data_editor(peak_editor_table, hide_index=True, use_container_width=True, disabled=disable_columns, key="peaks_editor")
                
                if st.button("Done editing"):
                    st.session_state.editing_peaks_table = False
                    st.rerun()

                if (edited_peaks_df != peak_editor_table).any().any():
                    if 'manual_peaks' not in st.session_state:
                        st.session_state.manual_peaks = {}

                    diffs = edited_peaks_df.ne(peak_editor_table)
                    edited_positions = np.where(diffs)
                    
                    validation_errors =[]

                    for row_idx, col_idx in zip(edited_positions[0], edited_positions[1]):
                        file_name = edited_peaks_df.iloc[row_idx]['File Name']
                        freq = edited_peaks_df.iloc[row_idx]['Frequency (Hz)']
                        db_column = 'Sound amplitude (dB SPL)' if level else 'Attenuation (dB)'
                        db = edited_peaks_df.iloc[row_idx][db_column]

                        column_name = edited_peaks_df.columns[col_idx]
                        new_value = edited_peaks_df.iloc[row_idx, col_idx]
                        
                        waveform_key = f"{file_name}_{freq}_{db}"
                        if waveform_key not in st.session_state.manual_peaks:
                            st.session_state.manual_peaks[waveform_key] = {}

                        st.session_state.manual_peaks[waveform_key][column_name]= new_value

                    for row_idx in edited_positions[0]:
                        file_name = edited_peaks_df.iloc[row_idx]['File Name']
                        freq = edited_peaks_df.iloc[row_idx]['Frequency (Hz)']
                        db_spl = edited_peaks_df.iloc[row_idx]['Sound amplitude (dB SPL)']
                        
                        # Get all peak latencies for this waveform (edited + original)
                        waveform_key = f"{file_name}_{freq}_{db_spl}"
                        latency_values = {}
                        
                        # Get latency columns and their current values
                        latency_cols = [col for col in edited_peaks_df.columns if 'latency' in col and 'Peak' in col]
                        for col in latency_cols:
                            latency_values[col] = edited_peaks_df.iloc[row_idx][col]
                        
                        # Check ordering
                        sorted_cols = sorted(latency_cols, key=lambda x: int(x.split()[1]))  # Sort by peak number
                        for i in range(len(sorted_cols) - 1):
                            current_val = latency_values[sorted_cols[i]]
                            next_val = latency_values[sorted_cols[i + 1]]
                            
                            if not pd.isna(current_val) and not pd.isna(next_val) and current_val >= next_val:
                                error_msg = f"{file_name}: {sorted_cols[i]} ({current_val}) should be < {sorted_cols[i + 1]} ({next_val})"
                                validation_errors.append(error_msg)
                                # st.toast(f"{error_msg}")
                    if validation_errors:
                        col2.error("**Warning:** Peak latencies are not in increasing order. Please check the edits.")
                
    else:
        tab2.write("Please upload files to analyze in the 'Data' tab.")

    st.sidebar.caption("[preprint](https://www.biorxiv.org/content/10.1101/2024.06.20.599815v2) | [github](https://github.com/ucsdmanorlab/abranalysis)")

if __name__ == "__main__":
    main()