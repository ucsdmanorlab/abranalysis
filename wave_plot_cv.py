import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.ndimage
from sklearn.model_selection import cross_val_score

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

def function_smoothing(x, sigma=3):
    if sigma is not None:  # Add a conditional check for None
        y = scipy.ndimage.gaussian_filter(x, sigma)
        return y
    else:
        return x 

def calculate_accuracy(data, window_size):
    smoothed_data = function_smoothing(data, sigma=window_size)
    
    # Dummy accuracy calculation (replace with your actual accuracy calculation)
    accuracy = np.mean(smoothed_data)
    
    return accuracy

def optimize_window_size(data, window_size_range):
    best_window_size = None
    best_accuracy = -np.inf

    for window_size in window_size_range:
        accuracy = calculate_accuracy(data, window_size)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_window_size = window_size
    
    return best_window_size

def plot_waves(dataframes, freq, db):
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

            # Cross-validation to optimize window size
            window_size_range = range(1, 11)  # Adjust the range as needed
            best_window_size = optimize_window_size(final, window_size_range)

            # Apply Gaussian smoothing with the optimized window size
            smoothed_final = function_smoothing(final, sigma=best_window_size)

            fig, ax = plt.subplots()  # Create a new figure for each plot
            ax.plot(smoothed_final)
            ax.plot(highest_peaks, smoothed_final[highest_peaks], "x")

            # Annotate the peaks with red color, smaller font, and closer to the peaks
            for peak in highest_peaks:
                ax.annotate(f'{smoothed_final[peak]:.5f}', (peak, smoothed_final[peak]), textcoords="offset points",
                            xytext=(0,3), ha='center', fontsize=8, color='red')

            ax.set_title(f'sheet: {filename} (Optimized Window Size: {best_window_size})')
            st.pyplot(fig)

# Streamlit UI
st.title("Wave Plotting App with Cross-Validated Smoothing")
st.sidebar.header("Enter Parameters")

# Frequency and dB level input fields
freq = st.sidebar.number_input("Enter Frequency (Hz)")
db = st.sidebar.number_input("Enter dB Level")

if st.button("Plot Waves"):
    plot_waves(dataframes, freq, db)