import datetime
import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from utils.calculate import display_threshold_table, display_peaks_table
from utils.plotting import *
from utils.processFiles import get_selected_data, process_uploaded_files_cached, db_column_name

import warnings
warnings.filterwarnings('ignore')

# TODO: consider converting freqs to kHz throughout for readability
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

def check_settings_and_clear_cache():
    calc_settings = {
        'time_scale': st.session_state.get('time_scale', 10.0),
        'multiply_y_factor': st.session_state.get('multiply_y_factor', 1.0),
        'units': st.session_state.get('units', 'Microvolts'),
        'return_units': st.session_state.get('return_units', 'Microvolts'),
        'calibration_levels': dict(st.session_state.get('calibration_levels', {}),),
        'peaks_below_thresh': st.session_state.get('peaks_below_thresh', False),
    }
    
    if 'previous_calc_settings' in st.session_state:
        if st.session_state.previous_calc_settings != calc_settings:
            st.warning("Settings have changed. Cleared calculated values.")
            if 'calculated_thresholds' in st.session_state:
                st.session_state.calculated_thresholds.clear()
            if 'calculated_waves' in st.session_state:
                st.session_state.calculated_waves.clear()
            clear_plots_and_tables()
            
            has_manual_thresholds = ('manual_thresholds' in st.session_state and st.session_state.manual_thresholds)
            has_manual_peaks = ('manual_peaks' in st.session_state and st.session_state.manual_peaks)
            
            if has_manual_thresholds or has_manual_peaks:
                with st.expander("🔧 Clear Manual Edits?", expanded=True):
                    st.warning("Manual edits detected. Do you want to clear them?")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    if has_manual_thresholds:
                        if col1.button("Clear Thresholds", use_container_width=True, type="primary", key="clear_thresh_settings"):
                            st.session_state.manual_thresholds.clear()
                            st.session_state.previous_calc_settings = calc_settings
                            st.success("Manual thresholds cleared!")
                            st.rerun()
                    
                    if has_manual_peaks:
                        if col2.button("Clear Peaks", use_container_width=True, type="primary", key="clear_peaks_settings"):
                            st.session_state.manual_peaks.clear()
                            if 'calculated_waves' in st.session_state:
                                st.session_state.calculated_waves.clear()
                            st.session_state.previous_calc_settings = calc_settings
                            st.success("Manual peaks cleared!")
                            st.rerun()
                    
                    if has_manual_thresholds and has_manual_peaks:
                        if col3.button("Clear All", use_container_width=True, type="primary", key="clear_all_settings"):
                            st.session_state.manual_thresholds.clear()
                            st.session_state.manual_peaks.clear()
                            if 'calculated_waves' in st.session_state:
                                st.session_state.calculated_waves.clear()
                            st.session_state.previous_calc_settings = calc_settings
                            st.success("All manual edits cleared!")
                            st.rerun()
                    
                    # Keep edits button
                    if st.button("Keep Manual Edits", use_container_width=True, type="secondary", key="keep_edits_settings"):
                        st.session_state.previous_calc_settings = calc_settings
                        st.info("Manual edits preserved.")
                        st.rerun()
            else:
                # No manual edits - just update settings
                st.session_state.previous_calc_settings = calc_settings
        else:
            # Settings haven't changed - show persistent manual edit management
            has_manual_thresholds = ('manual_thresholds' in st.session_state and st.session_state.manual_thresholds)
            has_manual_peaks = ('manual_peaks' in st.session_state and st.session_state.manual_peaks)
            
            if has_manual_thresholds or has_manual_peaks:
                with st.expander("🔧 Manual Edits Management", expanded=False):
                    st.info("Manual edits are currently active")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    if has_manual_thresholds:
                        if col1.button("Clear Thresholds", use_container_width=True, type="primary", key="clear_thresh_main"):
                            st.session_state.manual_thresholds.clear()
                            st.success("Manual thresholds cleared!")
                            st.rerun()
                    
                    if has_manual_peaks:
                        if col2.button("Clear Peaks", use_container_width=True, type="primary", key="clear_peaks_main"):
                            st.session_state.manual_peaks.clear()
                            if 'calculated_waves' in st.session_state:
                                st.session_state.calculated_waves.clear()
                            st.success("Manual peaks cleared!")
                            st.rerun()
                    
                    if has_manual_thresholds and has_manual_peaks:
                        if col3.button("Clear All", use_container_width=True, type="primary", key="clear_all_main"):
                            st.session_state.manual_thresholds.clear()
                            st.session_state.manual_peaks.clear()
                            if 'calculated_waves' in st.session_state:
                                st.session_state.calculated_waves.clear()
                            st.success("All manual edits cleared!")
                            st.rerun()
    else:
        # First time running - store settings
        st.session_state.previous_calc_settings = calc_settings

def get_current_plot_context():
    """Determine what type of plot is currently displayed"""
    if 'current_plot_filenames' not in st.session_state:
        return None
    
    filenames = st.session_state['current_plot_filenames']
    if not filenames:
        return None
    
    filename = filenames[0]  # Check first filename for pattern
    
    if "_stacked" in filename:
        return "stacked"
    # elif "3Dplot" in filename:
    #     return "3d_surface"
    # elif "IO_plot" in filename:
    #     return "io_curve"
    # elif any(db_str in filename for db_str in [f"{db}dB" for db in range(0, 121, 5)]):
    #     return "single_db"
    elif any(freq_str in filename for freq_str in [f"{freq/1000}kHz" for freq in range(125, 32001, 125)]):
        if len(st.session_state['current_plots']) == 1:
            return "single_tuple"  # Single plot = single frequency + single dB
        else:
            return "single_frequency"  # Multiple plots = single frequency, multiple dB
    else:
        return "single_tuple"  # Default fallback

def main():    
    if 'calibration_levels' not in st.session_state:
        st.session_state.calibration_levels = {}
    

    # Streamlit UI
    st.title("ABRA")
    tab1, tab2 = st.sidebar.tabs(["Data", "Plotting and Analysis"])
    
    uploaded_files = tab1.file_uploader("**Upload files to analyze:**", type=["csv", "arf", "asc", "tsv"], accept_multiple_files=True)
    tab1.markdown("<small><i>Note: All uploaded files are processed in memory only and are not retained by the application.</i></small>", unsafe_allow_html=True)
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
        validation_errors = []
        edits_made = False
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
        outputs = tab2.expander("**Visualization**", expanded=False)
        show_legend = outputs.toggle("Plot legends", True, key="show_legend")
        return_units = outputs.selectbox("Output units", options=['Microvolts', 'Nanovolts'], index=0, key="return_units")
        auto_y = outputs.toggle("Auto-scale Y", value=True, key="auto_y")
        if not auto_y:
            ycol1, ycol2 = outputs.columns(2)
            y_min = ycol1.number_input("Y min", value=-5000 if return_units == 'Nanovolts' else -5.0, key="y_min")
            y_max = ycol2.number_input("Y max", value=5000 if return_units == 'Nanovolts' else 5.0, key="y_max")
        plot_time_warped = outputs.toggle("Time warping", False, key="plot_time_warped")
        serif_font = outputs.toggle("Serif fonts", value=False, key="serif_font")
        #peak_settings = tab2.expander("**Peak & trough settings**", expanded=False)
        show_peaks = outputs.toggle("Show peaks/troughs", True)
        
        advanced_settings = tab2.expander("**Advanced settings**", expanded=False)
        multiply_y_factor = advanced_settings.number_input("Multiply Y values by factor", value=1.0, key="multiply_y_factor")
        vert_space = advanced_settings.number_input("Vertical space (for stacked curves)", value=10.0, min_value=0.0, step=1.0, key="vert_space")
        stacked_labels = advanced_settings.selectbox("Stacked labels position", options=["Left outside", "Right outside", "Right inside", "Off"], index=2)
        all_peaks = advanced_settings.toggle("ALL peaks/troughs in table (experimental)", value=False, key="all_peaks")
        peaks_below_thresh = advanced_settings.toggle("Peaks below threshold", value=False, key="peaks_below_thresh")

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
                try:
                    buffer = io.BytesIO()
                    fig.write_image(file=buffer, format="pdf")

                    st.download_button(
                        label="Download plot as PDF",
                        data=buffer,
                        file_name=st.session_state['current_plot_filenames'][i],
                        mime="application/pdf",
                        key=f'plot_download_{i}',
                        help="Download high-quality pdf plot"
                    )
                except Exception as e:
                    # Fallback to HTML for maximum compatibility
                    html_buffer = io.StringIO()
                    fig.write_html(html_buffer)
                    html_filename = st.session_state['current_plot_filenames'][i].replace('.pdf', '.html')

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.download_button(
                            label="Download as HTML",
                            data=html_buffer.getvalue(),
                            file_name=html_filename,
                            mime="text/html",
                            key=f'plot_download_html_{i}',
                            help="Download interactive plot - works in all browsers"
                        )
                    with col2:
                        with st.expander("PDF export unavailable - browser compatibility", expanded=False):
                            st.markdown("""
                            **For PDF exports:**
                            1. Install Chrome: [google.com/chrome](https://www.google.com/chrome/)
                            
                            **Alternative workflow:**
                            1. Download the HTML file
                            2. Open in browser → Print → Save as PDF
                            """)
                        
        lock_cols = ["File Name", "Frequency (Hz)", 'Sound amplitude (dB SPL)', 'Attenuation (dB)', 'Calibration Level (dB)']
        if 'current_table' in st.session_state and st.session_state['current_table'] is not None:
            metrics_table = st.session_state['current_table']
            st.dataframe(metrics_table, hide_index=True, use_container_width=True)
            
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
                    st.success(f"Set manual threshold for {file_name}, {freq} Hz to {threshold} dB. Choose plot option to refresh.")
        if 'peaks_table' in st.session_state and st.session_state['peaks_table'] is not None:
            peaks_table = st.session_state['peaks_table']
            
            # Initialize editing state
            if 'editing_peaks_table' not in st.session_state:
                st.session_state.editing_peaks_table = False
            
            if not st.session_state.editing_peaks_table:
                st.dataframe(peaks_table, hide_index=True, use_container_width=True)
                if 'peak_editor_table' in st.session_state and st.session_state.peak_editor_table is not None:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    if col1.button("Edit peaks table"):
                        st.session_state.editing_peaks_table = True
                        st.rerun()
                    
            else:
                col1, col2 = st.columns([1, 1])
                col1.warning("**Editing peaks mode**")

                peak_editor_table = st.session_state['peak_editor_table'] 
                disable_columns = [col for col in peak_editor_table.columns if 'latency' not in col]
                edited_peaks_df = st.data_editor(peak_editor_table, hide_index=True, use_container_width=True, disabled=disable_columns, key="peaks_editor")
                
                if st.button("Done editing"):
                    if (edited_peaks_df != peak_editor_table).any().any():
                        if 'manual_peaks' not in st.session_state:
                            st.session_state.manual_peaks = {}

                        diffs = edited_peaks_df.ne(peak_editor_table)
                        edited_positions = np.where(diffs)
                        
                        validation_errors = []
                        edits_made = False

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

                            # Store the manual edit
                            st.session_state.manual_peaks[waveform_key][column_name] = new_value
                            edits_made = True
                            
                        # check peak ordering:
                        for row_idx in set(edited_positions[0]):  
                            file_name = edited_peaks_df.iloc[row_idx]['File Name']
                            freq = edited_peaks_df.iloc[row_idx]['Frequency (Hz)']
                            db_spl = edited_peaks_df.iloc[row_idx]['Sound amplitude (dB SPL)']
                            
                            latency_values = {}
                            latency_cols = [col for col in edited_peaks_df.columns if 'latency' in col and 'Peak' in col]
                            for col in latency_cols:
                                latency_values[col] = edited_peaks_df.iloc[row_idx][col]
                            
                            sorted_cols = sorted(latency_cols, key=lambda x: int(x.split()[1]))
                            for i in range(len(sorted_cols) - 1):
                                current_val = latency_values[sorted_cols[i]]
                                next_val = latency_values[sorted_cols[i + 1]]
                                
                                if not pd.isna(current_val) and not pd.isna(next_val) and current_val >= next_val:
                                    error_msg = f"{file_name}: {sorted_cols[i]} ({current_val}) should be < {sorted_cols[i + 1]} ({next_val})"
                                    validation_errors.append(error_msg)
            
                        if validation_errors:
                            col2.error("**Warning:** Peak latencies are not in increasing order!")
                            for error in validation_errors:
                                st.error(error)

                    if edits_made:
                        plot_type = get_current_plot_context()                
                        try:
                            if plot_type == "single_tuple":
                                updated_plots = [plot_waves_single_tuple(selected_dfs, selected_files,freq, db, show_peaks=show_peaks)]
                                st.session_state['peaks_table'] = display_peaks_table(selected_dfs, selected_files, [freq], [db], return_threshold=True, return_nas=True)
                                st.session_state['peak_editor_table'] = display_peaks_table(selected_dfs, selected_files, [freq], [db], return_threshold=True, return_nas=True, editable=True)
                                
                            elif plot_type == "single_frequency":
                                updated_plots = plot_waves_single_frequency(selected_dfs, selected_files, freq, plot_time_warped=plot_time_warped, show_peaks=show_peaks)
                                st.session_state['peaks_table'] = display_peaks_table(selected_dfs, selected_files, [freq], distinct_dbs)
                                st.session_state['peak_editor_table'] = display_peaks_table(selected_dfs, selected_files, [freq], distinct_dbs, return_threshold=True, return_nas=True, editable=True)
                            
                            st.session_state['current_plots'] = updated_plots
                            
                        except Exception as e:
                            st.error(f"Error updating plots: {e}")
                    st.session_state.editing_peaks_table = False
                    st.rerun()
            
    else:
        tab2.write("Please upload files to analyze in the 'Data' tab.")

    st.sidebar.caption("[preprint](https://www.biorxiv.org/content/10.1101/2024.06.20.599815v2) | [github](https://github.com/ucsdmanorlab/abranalysis)")
    st.caption("_ABRA uses automated algorithms for ABR analysis, which may occasionally produce incorrect labels or thresholds. Users should visually verify results and apply expert judgment before drawing conclusions._")

if __name__ == "__main__":
    main()            