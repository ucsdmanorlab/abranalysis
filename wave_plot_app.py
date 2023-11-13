import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Read each CSV file in the current_directory and store it in dataframes
current_directory = os.getcwd()
csv_files = glob.glob(os.path.join(current_directory, 'ABR_exported_files/*.csv'))

dataframes = {}
for file in csv_files:
    filename = os.path.splitext(os.path.basename(file))[0]
    if pd.read_csv(file).shape[1] > 1:
        dataframes[filename] = pd.read_csv(file)
    else:
        dataframes[filename] = pd.read_csv(file, skiprows=2)

def plot_waves(dataframes, freq=16000, db=90):
    for filename, df in dataframes.items():
        khz = df[df['Freq(Hz)'] == freq]
        dbkhz = khz[khz['Level(dB)'] == db]
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
            for peak in highest_peaks:
                ax.annotate(f'{final[peak]:.3f}', (peak, final[peak]),
                            textcoords="offset points", xytext=(0,3),
                            ha='center', fontsize=8, color='red')

            ax.set_title(f'sheet: {filename}')
            st.pyplot(fig)

# Streamlit UI
st.title("Wave Plotting App")
st.sidebar.header("Enter Parameters")

# Frequency and dB level input fields
freq = st.sidebar.number_input("Enter Frequency (Hz)")
db = st.sidebar.number_input("Enter dB Level")

if st.button("Plot Waves"):
    plot_waves(dataframes, freq, db)