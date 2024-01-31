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
import warnings                               
warnings.filterwarnings('ignore')

def plot_scatter_waves(df, freq, db, background_curves=False, smoothing_method='None', sigma=3, n=15):
    fig = go.Figure()
    khz = df[(df['Freq(Hz)'].astype(float) == freq) & (df['Level(dB)'].astype(float) == db)]
    
    if not khz.empty:
        index = khz.index.values[0]
        final = df.iloc[index, 48:]
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
    original_waveform = df.iloc[index, 48:]
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
    unique_dbs = sorted(df['Level(dB)'].unique())

    wave_colors = [f'rgb(255, {b}, {b})' for b in np.linspace(255, 100, len(unique_dbs))]

    waves_array = []  # Array to store all waves

    for db in sorted(df['Level(dB)'].unique()):
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.iloc[index, 48:]
            final = pd.to_numeric(final, errors='coerce')

            if multiply_y_factor != 1:
                y_values = final * multiply_y_factor
            else:
                y_values = final

            waves_array.append(y_values.to_list())  # Append the current wave to the array
    
    #waves_array = np.array(waves_array)
    waves_array = np.array([wave[:-1] for wave in waves_array])
    
    # Optionally apply time warping to all waves in the array
    if plot_time_warped:
        time = np.linspace(0, 10, waves_array.shape[1])
        obj = fs.fdawarp(np.array(waves_array).T, time)
        obj.srsf_align(parallel=True)
        waves_array = obj.fn.T  # Use the time-warped curves

    # Plot all waves in the array
    for i, (db, waves) in enumerate(zip(sorted(df['Level(dB)'].unique()), waves_array)):
        fig.add_trace(go.Scatter(x=np.linspace(0,10, waves_array.shape[1]), y=waves, mode='lines', name=f'dB: {db}', line=dict(color=wave_colors[i])))

    fig.update_layout(title=f'{uploaded_files[0].name} - Frequency: {freq} Hz', xaxis_title='Time (ms)', yaxis_title='Voltage (mV)')
    fig.update_layout(annotations=annotations)
    fig.update_layout(yaxis_range=[y_min, y_max])

    return fig

def plot_waves_single_db(df, db, y_min, y_max):
    fig = go.Figure()

    for freq in sorted(df['Freq(Hz)'].unique()):
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.iloc[index, 48:]
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
            final = df.iloc[index, 48:]
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
    wave_colors = [f'rgb(255, {b}, {b})' for b in np.linspace(0, 0, len(db_levels))]
    connecting_line_color = 'rgba(0, 255, 0, 0.3)'

    for db in db_levels:
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        
        if not khz.empty:
            index = khz.index.values[0]
            final = df.iloc[index, 48:]
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
        final = df.iloc[index, 48:]
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
                'Metric': ['First Peak Amplitude (mV)', 'Latency to First Peak (ms)', 'Amplitude Ratio (Peak1/Peak4)'],
                'Value': [first_peak_amplitude, latency_to_first_peak, amplitude_ratio]
            })

            st.table(metrics_table)

def display_metrics_table_all_db(df, freq, db_levels, baseline_level):
    metrics_data = {'dB Level': [], 'First Peak Amplitude (mV)': [], 'Latency to First Peak (ms)': [], 'Amplitude Ratio (Peak1/Peak4)': []}
    
    for db in db_levels:
        khz = df[(df['Freq(Hz)'] == freq) & (df['Level(dB)'] == db)]
        if not khz.empty:
            index = khz.index.values[0]
            final = df.iloc[index, 48:]
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


# Streamlit UI
st.title("Wave Plotting App")
st.sidebar.header("Upload CSV File")
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=True)

annotations = []

if uploaded_files:
    dfs = []
    
    for file in uploaded_files:
        temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.read())

        if pd.read_csv(temp_file_path).shape[1] > 1:
            df = pd.read_csv(temp_file_path)
        else:
            df = pd.read_csv(temp_file_path, skiprows=2)
            
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
    
    #if st.sidebar.button("Plot Waves with Cubic Spline"):
    #    fig = plotting_waves_cubic_spline(df, freq, db)
    #    fig.update_layout(yaxis_range=[y_min, y_max])
    #    st.plotly_chart(fig)

    if st.sidebar.button("Plot 3D Surface"):
        fig_3d_surface = plot_3d_surface(df, freq, y_min, y_max)
        st.plotly_chart(fig_3d_surface)
    
    #st.markdown(get_download_link(fig), unsafe_allow_html=True)
