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

def plot_waves(ax, df, freq, db, smoothing_method='None', sigma=3, n=15):
    khz = df[df['Freq(Hz)'].astype(float) == freq]
    dbkhz = khz[khz['Level(dB)'].astype(float) == db]
    if not dbkhz.empty:
        index = dbkhz.index.values[0]
        final = df.iloc[index, 48:]
        final = pd.to_numeric(final, errors='coerce')

        # Find highest peaks separated by at least n data points
        n = 20
        peaks, _ = find_peaks(final, distance=n)
        highest_peaks = peaks[np.argsort(final[peaks])[-5:]]

        fig, ax = plt.subplots()  # Create a new figure for each plot
        ax.plot(final)
        ax.plot(highest_peaks, final[highest_peaks], "x")

        # Annotate the peaks with red color, smaller font, and closer to the peaks
        for peak in highest_peaks:
            ax.annotate(f'{final[peak]:.2f}', (peak, final[peak]), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='red')

        ax.set_title('Data from User Upload')
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

    # Frequency and dB level input fields
    freq = st.sidebar.number_input("Enter Frequency (Hz)", min_value=float(df['Freq(Hz)'].min()), max_value=float(df['Freq(Hz)'].max()), value=float(df['Freq(Hz)'].min()))
    db = st.sidebar.number_input("Enter dB Level", min_value=float(df['Level(dB)'].min()), max_value=float(df['Level(dB)'].max()), value=float(df['Level(dB)'].min()))

    if st.sidebar.button("Plot Waves"):
        plot_waves(df, freq, db)
