import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import os
import tempfile
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

gt_data = {
    'B2_1283_baseline': {
        4000: {
            90: 0.0189,
            70: 0.00494
        },
        8000: {
            90: 0.01488,
            70: 0.01046
        },
        16000: {
            90: 0.02141,
            70: 0.01074
        },
        24000: {
            90: 0.02374,
            70: 0.01365
        },
        32000: {
            90: 0.02074,
            70: 0.01269
        }
    },
    'B2_1288_baseline': {
        4000: {
            90: 0.01338,
            70: 0.00057
        },
        8000: {
            90: 0.01093,
            70: 0.0014
        },
        16000: {
            90: 0.01667,
            70: 0.01044
        },
        24000: {
            90: 0.01504,
            70: 0.00747
        },
        32000: {
            90: 0.01421,
            70: 0.00444
        }
    },
    'B1_1282_baseline': {
        4000: {
            90: 0.02278,
            70: 0.0029
        },
        8000: {
            90: 0.02173,
            70: 0.0059
        },
        16000: {
            90: 0.03456,
            70: 0.02531
        },
        24000: {
            90: 0.03234,
            70: 0.01168
        },
        32000: {
            90: 0.02777,
            70: 0.00671
        }
    }
}

def plot_waves(df, freq, db, smoothing_method='None', sigma=3, n=15):
    khz = df[df['Freq(Hz)'].astype(float) == freq]
    dbkhz = khz[khz['Level(dB)'].astype(float) == db]
    if not dbkhz.empty:
        index = dbkhz.index.values[0]
        final = df.iloc[index, 48:]
        final = pd.to_numeric(final, errors='coerce')

        # Find highest peaks separated by at least n data points
        peaks, _ = find_peaks(final, distance=n)
        highest_peaks = peaks[np.argsort(final[peaks])[-5:]]

        fig, ax = plt.subplots()  # Create a new figure for each plot
        ax.plot(final)
        ax.plot(highest_peaks, final[highest_peaks], "x")

        # Annotate the peaks with red color, smaller font, and closer to the peaks
        for peak in highest_peaks:
            ax.annotate(f'{final[peak]:.4f}', (peak, final[peak]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='red')

        ax.set_title('Data from User Upload')

        xticks = np.arange(0, len(final), 20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        st.pyplot(fig)

def plotting_waves_cubic_spline(df, freq=16000, db=90, n=15):
    khz = df[df['Freq(Hz)'] == freq]
    dbkhz = khz[khz['Level(dB)'] == db]
    index = dbkhz.index.values[0]
    original_waveform = df.iloc[index, 48:]
    original_waveform = pd.to_numeric(original_waveform, errors='coerce')[:-1]

    # Apply cubic spline interpolation
    smooth_time = np.linspace(0, len(original_waveform) - 1, 100)
    cs = CubicSpline(np.arange(len(original_waveform)), original_waveform)
    smooth_amplitude = cs(smooth_time)

    # Find highest peaks separated by at least n data points in the smoothed curve
    peaks, _ = find_peaks(smooth_amplitude, distance=n)
    highest_peaks = peaks[np.argsort(smooth_amplitude[peaks])[-5:]]

    loss_value = None
    if highest_peaks.size > 0:  # Check if highest_peaks is not empty
        first_peak_value = smooth_amplitude[np.sort(highest_peaks)[0]]
        loss_value = first_peak_value  # Modify as needed

    fig, ax = plt.subplots()

    # Plot the original ABR waveform
    ax.plot(np.arange(len(original_waveform)), original_waveform, label='Original ABR', alpha=0.8)

    # Plot the cubic spline interpolation
    ax.plot(smooth_time, smooth_amplitude, label='Cubic Spline Interpolation')

    if highest_peaks.size > 0:
        first_peak = np.sort(highest_peaks)[0]
        ax.plot([smooth_time[first_peak], smooth_time[first_peak]], [smooth_amplitude[first_peak], np.nan], "--", color='gray')
        ax.plot(smooth_time[first_peak], np.nan, "o", color='blue', alpha=0.5)

    x_ticks = np.arange(0, len(original_waveform), 20)
    ax.set_xticks(x_ticks)

    for peak in highest_peaks:
        ax.annotate(f'{smooth_amplitude[peak]:.4f}', (smooth_time[peak], smooth_amplitude[peak]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='red')

    ax.set_title('Data from User Upload')
    ax.plot(smooth_time[highest_peaks], smooth_amplitude[highest_peaks], "x")

    ax.legend()
    st.pyplot(fig)

def plotting_waves_gauss(df, freq=16000, db=90, n=15, sigma=3):
    khz = df[df['Freq(Hz)'] == freq]
    dbkhz = khz[khz['Level(dB)'] == db]
    index = dbkhz.index.values[0]
    original_waveform = df.iloc[index, 48:]
    original_waveform = pd.to_numeric(original_waveform, errors='coerce')

    # Apply Gaussian smoothing to the original ABR waveform
    smoothed_waveform = gaussian_filter1d(original_waveform, sigma=sigma)

    # Find highest peaks separated by at least n data points in the original curve
    original_peaks, _ = find_peaks(original_waveform, distance=n)
    highest_original_peaks = original_peaks[np.argsort(original_waveform[original_peaks])[-5:]]

    # Find highest peaks separated by at least n data points in the smoothed curve
    smoothed_peaks, _ = find_peaks(smoothed_waveform, distance=n)
    highest_smoothed_peaks = smoothed_peaks[np.argsort(smoothed_waveform[smoothed_peaks])[-5:]]

    loss_value = None
    if highest_smoothed_peaks.size > 0:  # Check if highest_smoothed_peaks is not empty
        first_peak_value = smoothed_waveform[np.sort(highest_smoothed_peaks)[0]]
        # Note: Ground truth data is not considered in this example

    fig, ax = plt.subplots()

    # Plot the original ABR waveform
    ax.plot(original_waveform, label='Original ABR')

    # Plot the smoothed ABR waveform
    ax.plot(smoothed_waveform, label=f'Gaussian Smoothed (sigma={sigma})')

    if highest_original_peaks.size > 0:  # Check if highest_original_peaks is not empty
        first_original_peak = np.sort(highest_original_peaks)[0]
        ax.plot(first_original_peak, original_waveform[first_original_peak], "o", color='red', label='Original Peaks', alpha=0.5)

    if highest_smoothed_peaks.size > 0:  # Check if highest_smoothed_peaks is not empty
        first_smoothed_peak = np.sort(highest_smoothed_peaks)[0]
        ax.plot(first_smoothed_peak, smoothed_waveform[first_smoothed_peak], "x", label='Smoothed Peaks', color='blue')

    x_ticks = np.arange(0, len(original_waveform), 20)
    ax.set_xticks(x_ticks)

    ax.set_title('Data from User Upload')
    ax.legend()
    st.pyplot(fig)

# Streamlit UI
st.title("Wave Plotting App")
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    if pd.read_csv(temp_file_path).shape[1] > 1:
        df = pd.read_csv(temp_file_path)
    else:
        df = pd.read_csv(temp_file_path, skiprows=2)

    st.sidebar.subheader("Enter Parameters")

    # Frequency and dB level input fields as sliders with discrete options
    freq_options = sorted(df['Freq(Hz)'].unique())
    freq = st.sidebar.slider("Select Frequency (Hz)", min_value=min(freq_options), max_value=max(freq_options), value=min(freq_options), step=4000.0)

    # dB Level dropdown options
    db_options = sorted(df['Level(dB)'].unique())
    db = st.sidebar.slider("Select dB Level", min_value=min(db_options), max_value=max(db_options), value=min(db_options), step=5.0)


    if st.sidebar.button("Plot Waves"):
        plot_waves(df, freq, db)

    if st.sidebar.button("Plot Waves with Cubic Spline"):
        plotting_waves_cubic_spline(df, freq, db)
    
    if st.sidebar.button("Plot Waves with Gaussian Smoothing"):
        plotting_waves_gauss(df, freq, db)
