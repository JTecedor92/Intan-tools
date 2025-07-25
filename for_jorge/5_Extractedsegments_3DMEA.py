#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:34:21 2025

@author: jameslim

This script is for analyzing the saved extracted_segments that are already 
filtered in Mehdi's visualization script. This extracted segments array should have
the segments of data for each stim trial'. Arrays may be filtered to visualize 
action potentials (>300Hz) or LFPs (bandpassing anywhere between 0-300Hz) after 
stimulation.

'extracted_segments_f' is filtered according the graphs it represents.
300-3000Hz

as of Aug 27, all LFP extracted_segments arrays are downsampled to 2000Hz before being loaded
into this script. I might integrate an extracted array that is not downsampled in order to visualize 
high pass filtered >300Hz in the 10 second window

figure 4 of first manuscript
"""


import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D

# from intanutil.header import (read_header,
#                               header_to_result)
# from intanutil.data import (calculate_data_size,
#                             read_all_data_blocks,
#                             check_end_of_file,
#                             parse_data,
#                             data_to_result)
# from intanutil.filter import apply_notch_filter

# In[]:
Fs = 2000
# Create a Tkinter root window, which is required for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the .npy file
file_path = filedialog.askopenfilename(
    title="Select .npy file", 
    filetypes=[("NumPy files", "*.npy")], 
    initialdir="."  # Optional: set the initial directory
)

# Check if a file was selected
if file_path:
    # Load the selected .npy file
    extracted_segments = np.load(file_path)
    print("File loaded successfully.")
    print("Data:", extracted_segments)
else:
    print("No file selected.")

# In[]:
#Fs = 30000
# Create a Tkinter root window, which is required for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the .npy file
file_path = filedialog.askopenfilename(
    title="Select before tissue .npy file", 
    filetypes=[("NumPy files", "*.npy")], 
    initialdir="."  # Optional: set the initial directory
)

# Check if a file was selected
if file_path:
    # Load the selected .npy file
    extracted_segments_ctrl = np.load(file_path)
    print("File loaded successfully.")
    print("Data:", extracted_segments)
else:
    print("No file selected.")   

# In[]
"""load corresponding impedance.csv file for extracted_segments array"""
Fs = 2000
# Create a Tkinter root window, which is required for the file dialog
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select the .npy file
file_path = filedialog.askopenfilename(
    title="Select .csv file", 
    filetypes=[("csv files", "*.csv")], 
    initialdir="."  # Optional: set the initial directory
)
# Check if a file was selected
if file_path:
    # Load the CSV file into a NumPy array
    impedance_info = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    print("File loaded successfully.")
    print("Data:", impedance_info)
    print('Channels and Headers: ', impedance_info.shape)
else:
    print("No file selected.")

# In[] load impedance file with pandas
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

# Create a Tkinter root window (hidden)
root = tk.Tk()
root.withdraw()  

# Open file dialog to select the CSV file
file_path = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=[("CSV files", "*.csv")],
    initialdir="."
)

# Check if a file was selected
if file_path:
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Extract channel names (2nd column, index 1)
    channel_names = df.iloc[:, 1].values  

    # Extract impedance measurements (5th column, index 4)
    impedance_values = df.iloc[:, 4].values  
    
    print("File loaded successfully.")
    print("Channels:", channel_names)
    print("Impedance Values:", impedance_values)
else:
    print("No file selected.")

# In[ ]: FILTER DEFINITIONS

def LP_IIR(signal, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.iirfilter(order, normal_cutoff,
                                  btype='lowpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def HP_IIR(signal, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.iirfilter(order, normal_cutoff,
                                  btype='highpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def BP_IIR(signal, cutoff1, cutoff2, fs, order):
    nyq = 0.5 * fs
    normal_cutoff1 = cutoff1 / nyq
    normal_cutoff2 = cutoff2 / nyq

    b, a = scipy.signal.iirfilter(order, [normal_cutoff1, normal_cutoff2],
                                  btype='bandpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def BS_IIR(signal, cutoff1, cutoff2, fs, order):
    nyq = 0.5 * fs
    normal_cutoff1 = cutoff1 / nyq
    normal_cutoff2 = cutoff2 / nyq
    b, a = scipy.signal.iirfilter(order, [normal_cutoff1, normal_cutoff2],
                                  btype='bandstop', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def AP_IIR(signal, fs, order):
    nyq = 0.5 * fs
    cutoff1 = 300
    cutoff2 = 3000
    normal_cutoff1 = cutoff1 / nyq
    normal_cutoff2 = cutoff2 / nyq

    b, a = scipy.signal.iirfilter(order, [normal_cutoff1, normal_cutoff2],
                                  btype='bandpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter


def LFP_IIR(signal, fs, order):
    nyq = 0.5 * fs
    cutoff = 1000

    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.iirfilter(order, normal_cutoff,
                                  btype='lowpass', analog=False, ftype='butter')
    # print(b, a, sep="\n")
    y_lfilter = scipy.signal.lfilter(b, a, signal)

    return y_lfilter

# In[ ]: band pass extracted_segments array into desired low frequency bands

def bandpass_extracted_segments(filtered_segments,fs,order=3):
    '''
    Create filtered_segments arrays dedicated to a certain low frequency band
    Parameters:
        - array: filtered_segments array [trials, channels, samples]
        - sampling frequency: fs
        - order: filter order
        
    Returns:
        - filtered_segments_delta
        - filtered_segments_theta
        - filtered_segments_beta
        - filtered_segments_gamma
        - filtered_segments_ripple
    '''
    # Delta
    cutoff_delta = 1.5  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_delta = 4
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_delta = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_delta[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_delta, cutoff1_delta, fs, order)

    # Theta
    cutoff_theta = 4  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_theta = 12
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_theta = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_theta[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_theta, cutoff1_theta, fs, order)

    # Beta
    cutoff_beta = 12  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_beta = 30
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_beta = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_beta[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_beta, cutoff1_beta, fs, order)

    # Gamma (30-100Hz)
    cutoff_gamma = 30  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_gamma = 100
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_gamma = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_gamma[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_gamma, cutoff1_gamma, fs, order)

    # Ripple (100-250Hz)
    cutoff_ripple = 100  # Cutoff frequency for lowpass filter (Hz)
    cutoff1_ripple = 250
    order = 3  # Filter order

    # Apply the filter to each channel of the data
    f_segments_ripple = np.zeros_like(filtered_segments)
    for i in range(filtered_segments.shape[1]):  # Loop through each channel
        f_segments_ripple[:, i, :] = BP_IIR(
            filtered_segments[:, i, :], cutoff_ripple, cutoff1_ripple, fs, order)

    
    return f_segments_delta, f_segments_theta, f_segments_beta, f_segments_gamma, f_segments_ripple
# In[]
"""Use bandpass extracted segments function on both filtered segments 1 and 2"""
# Example Usage:
array1 = extracted_segments
fs = 2000
f_segments_delta1, f_segments_theta1, f_segments_beta1, f_segments_gamma1, f_segments_ripple1 = bandpass_extracted_segments(array1,fs,order=3)
print('Function Complete!')

array2 = extracted_segments_ctrl
fs = 2000
f_segments_delta2, f_segments_theta2, f_segments_beta2, f_segments_gamma2, f_segments_ripple2 = bandpass_extracted_segments(array2,fs,order=3)
print('Function Complete!')
# In[ ]: band pass extracted_segments array into desired low frequency bands

def plot_f_segments(ch, start, stop, f_segments_delta, f_segments_theta, f_segments_beta, f_segments_gamma, f_segments_ripple):
    '''
    Plot frequency bands from f_segments_arrays
    Parameters:
        - 
    Returns:
        - time_axis
    '''
    # Create time axis
    time_axis = np.arange(f_segments_beta.shape[2])/fs
    
    # Get shape of filtered_segments
    trials, channels, samples = f_segments_beta.shape
    print('trials, channels, samples: ', f_segments_beta.shape)

    # Plot individual trial responses and display each frequency band
    for trial in range(trials):
        #plt.plot(time_axis[start:stop], f_segments_delta[trial,
        #          ch, start:stop], label='delta(1-4Hz)')
        # plt.plot(time_axis[start:stop], f_segments_theta[trial,
        #           ch, start:stop], label='theta(4-10Hz)')
        # plt.plot(time_axis[start:stop], f_segments_beta[trial,
        #           ch, start:stop], label='beta(12-30Hz)')
        # plt.plot(time_axis[start:stop], f_segments_gamma[trial,
        #           ch, start:stop], label='gamma(30-100Hz)')
        # plt.plot(time_axis[start:stop], f_segments_ripple[trial,
        #           ch,start:stop], label='ripple(100-250Hz)')
        plt.plot(time_axis[start:stop], extracted_segments[trial,
                    ch, start:stop], label='all(0-300Hz)', alpha=0.5)
        plt.xlabel('Time(s)')
        plt.ylabel('uV')
        plt.ylim(-500, 500)
        plt.axvline(x=stim_time, color='black',
                    linestyle='--', linewidth=2, label='STIM')
        plt.title(f'Frequency bands channel {ch} - Trial {trial+1} response')
        plt.grid(True)
        plt.legend(loc='upper right', fontsize='x-small', ncol=2)
        plt.show()
    
    return time_axis

# Example Usage:
ch = 34
start = int(0*Fs)
stop = int(10*Fs)
stim_time = 2
time_axis = plot_f_segments(ch, start, stop,
        f_segments_delta1, f_segments_theta1, f_segments_beta1, 
        f_segments_gamma1, f_segments_ripple1)
print('plot_f_segments COMPLETE!')

ch = 34
start = int(0*Fs)
stop = int(10*Fs)
stim_time = 2
time_axis = plot_f_segments(ch, start, stop,
        f_segments_delta2, f_segments_theta2, f_segments_beta2, 
        f_segments_gamma2, f_segments_ripple2)
print('plot_f_segments COMPLETE!')

# In[]
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
from scipy.signal import welch

def compute_average_power(extracted_segments, start_time, end_time, fs=2000):
    """
    Computes the average power for each channel across all trials in the extracted_segments array within a selected time window.

    Parameters:
    - extracted_segments: numpy array of shape (trials, channels, samples)
    - start_time: start time of the window in seconds
    - end_time: end time of the window in seconds
    - fs: sampling frequency (default: 2 kHz)

    Returns:
    - average_power: numpy array of average power for each channel (shape: channels,)
    """
    num_trials, num_channels, num_samples = extracted_segments.shape
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    average_power = np.zeros(num_channels)  # Store average power per channel

    for channel in range(num_channels):
        psd_list = []
        for trial in range(num_trials):
            f, Pxx = welch(extracted_segments[trial, channel, start_sample:end_sample], fs=fs)
            psd_list.append(Pxx)  # Store PSD for this trial

        # Compute mean power across trials and frequency bins
        # average_power[channel] = np.mean([np.mean(Pxx) for Pxx in psd_list])  
        average_power[channel] = np.mean(psd_list)  


    return average_power, psd_list


def get_channel_from_name(name):
    """
    Extracts the channel number from the filtered_segments variable name.

    Parameters:
    - name: string, the name of the filtered_segments variable (e.g., 'filtered_segments_25')

    Returns:
    - int: extracted channel number
    """
    # Extract the numeric part from the variable name (assumes the format 'filtered_segments_<channel>')
    return int(name.split('_')[-1])

def plot_average_power(average_power, layout, selected_channel, title='Average Power Visualization', vmin=None, vmax=None):
    """
    Plots the average power of each channel within a specified layout, highlighting the selected channel in red.

    Parameters:
    - average_power: numpy array of average power for each channel (shape: channels,)
    - layout: list of lists specifying the channel layout
    - selected_channel: channel index to be highlighted in red
    - title: string, the title of the plot
    - vmin, vmax: normalization range for the heatmap color bar
    """
    # Create a mapping from channel names to average power
    channel_map = {}

    selected_coords = None  # To store the coordinates of the selected channel

    for i, row in enumerate(layout):
        # annotation_row = []
        for j, channel_name in enumerate(row):
            channel_index = int(channel_name[1:])  # Extract channel number from name
            channel_map[channel_name] = average_power[channel_index]
            # annotation_row.append(channel_name)
            # annotations.append(annotation_row)
            # # Check if this is the selected channel to highlight
            # if channel_index == selected_channel:
            #     selected_coords = (i, j)

    # Prepare data for heatmap
    heatmap_data = np.zeros((len(layout), len(layout[0])))
    for i, row in enumerate(layout):
        for j, channel_name in enumerate(row):
            heatmap_data[i, j] = channel_map[channel_name]

    # Plot the heatmap with consistent normalization using vmin and vmax
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        # annot=annotations,
        fmt='s',
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap='viridis',
        cbar=True,
    )

    # Highlight the selected channel in red by overlaying a separate heatmap
    if selected_coords:
        mask = np.ones_like(heatmap_data, dtype=bool)  # Mask everything
        mask[selected_coords] = False  # Unmask the selected channel
        sns.heatmap(
            heatmap_data,
            annot=layout,
            fmt='',
            cmap=ListedColormap(['red']),
            cbar=False,
            mask=mask,  # Mask everything except the highlighted channel
            linewidths=0.5,
        )

    # Remove tick marks
    plt.xticks([])
    plt.yticks([])

    plt.title(title)
    plt.show()

# Example usage:
# filtered_segments_25 = np.random.rand(10, 64, 6000)  # Example data for channel 25
# filtered_segments_27 = np.random.rand(10, 64, 6000)  # Example data for channel 27
sampling_rate = 2000  # Define the sampling rate (Hz)

# Compute average power for each set
# average_power, psd_list = compute_average_power(extracted_segments, start_time=3, end_time=7, fs=sampling_rate)
# average_power_2, psd_list = compute_average_power(extracted_segments_ctrl, start_time=3, end_time=7, fs=sampling_rate)
average_power, psd_list = compute_average_power(f_segments_delta1, start_time=3, end_time=5, fs=sampling_rate)
average_power_2, psd_list = compute_average_power(f_segments_delta2, start_time=3, end_time=5, fs=sampling_rate)
# average_power, psd_list = compute_average_power(f_segments_theta1, start_time=3, end_time=7, fs=sampling_rate)
# average_power_2, psd_list = compute_average_power(f_segments_theta2, start_time=3, end_time=7, fs=sampling_rate)
# average_power, psd_list = compute_average_power(f_segments_gamma1, start_time=3, end_time=4, fs=sampling_rate)
# average_power_2, psd_list = compute_average_power(f_segments_gamma2, start_time=3, end_time=4, fs=sampling_rate)

# # Extract channel numbers from the names
ch1 = get_channel_from_name('filtered_segments_02')
ch2 = get_channel_from_name('filtered_segments_021')

# Find the global min and max for consistent normalization
vmin = average_power.min()
vmax = average_power.max()

# Define the channel layout as provided
"""Neuronexus channel layout"""
layout = [
    ['A7', 'A6', 'A14', 'A20', 'B24', 'B25', 'B17', 'B11'],
    ['A0', 'A4', 'A13', 'A22', 'B31', 'B27', 'B18', 'B9'],
    ['A23', 'A17', 'A27', 'A11', 'B6', 'B30', 'B10', 'B20'],
    ['A31', 'A19', 'A12', 'A18', 'B0', 'B2', 'B19', 'B13'],
    ['A28', 'A2', 'A29', 'A15', 'B3', 'B29', 'B12', 'B16'],
    ['A5', 'A21', 'A1', 'A25', 'B26', 'B4', 'B14', 'B8'],
    ['A24', 'A3', 'A10', 'A16', 'B7', 'B28', 'B21', 'B15'],
    ['A26', 'A30', 'A8', 'A9', 'B5', 'B1', 'B23', 'B22']
] # 

# """MED64 channel layout"""
# layout = [
#             ['A22', 'B18', 'B30', 'B24', 'B1',  'B3',  'B5',  'B23'],
#             ['A23', 'B20', 'B16', 'B26', 'B25', 'B27', 'B22', 'A24'],
#             ['B21', 'B14', 'B15', 'B28', 'B2',  'B4',  'A0',  'A1' ],
#             ['B19', 'B12', 'B13', 'B11', 'B0',  'A25', 'A2',  'A3' ],
#             ['B10', 'B17', 'B9',  'A14', 'A26', 'A27', 'A5',  'A4' ],
#             ['B8',  'B31', 'A30', 'A20', 'A12', 'A29', 'A7',  'A6' ],
#             ['B7',  'B6',  'A16', 'A21', 'A13', 'A11', 'A31', 'A8' ],
#             ['B29', 'A28', 'A18', 'A15', 'A19', 'A17', 'A10', 'A9' ]
#         ] 
# annotations = np.empty_like(layout, dtype=object)  # Create an array of the same shape as layout
# Plot the average power for each channel according to the layout with consistent color scale
start_time = 3
end_time = 5
plot_average_power(average_power, layout, selected_channel=ch1, title=f'Average Power - Stimulation at ch{ch1}', vmin=vmin, vmax=vmax)
plot_average_power(average_power_2, layout, selected_channel=ch2, title=f'Average Power - Stimulation at ch{ch2}', vmin=vmin, vmax=vmax)

# In[] figure 4 for manuscript
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

def compute_average_power(extracted_segments, start_time, end_time, fs=2000):
    num_trials, num_channels, num_samples = extracted_segments.shape
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    average_power = np.zeros(num_channels)

    for channel in range(num_channels):
        psd_list = []
        for trial in range(num_trials):
            f, Pxx = welch(extracted_segments[trial, channel, start_sample:end_sample], fs=fs)
            psd_list.append(Pxx)
        average_power[channel] = np.mean(psd_list)

    return average_power, psd_list

def get_channel_from_name(name):
    return int(name.split('_')[-1])

def plot_average_power(average_power, layout, selected_channel, title='', vmin=None, vmax=None, ax=None, show_colorbar=True):
    channel_map = {}
    selected_coords = None

    for i, row in enumerate(layout):
        for j, channel_name in enumerate(row):
            channel_index = int(channel_name[1:])
            channel_map[channel_name] = average_power[channel_index]
            if channel_index == selected_channel:
                selected_coords = (i, j)

    heatmap_data = np.zeros((len(layout), len(layout[0])))
    for i, row in enumerate(layout):
        for j, channel_name in enumerate(row):
            heatmap_data[i, j] = channel_map[channel_name]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        heatmap_data,
        ax=ax,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap='viridis',
        cbar=show_colorbar,
        xticklabels=False,
        yticklabels=False,
    )

    # Highlight the selected channel in red with a rectangle
    for i, row in enumerate(layout):
        for j, channel_name in enumerate(row):
            channel_index = int(channel_name[1:])
            if channel_index == selected_channel:
                ax = plt.gca()
                rect = plt.Rectangle((j, i), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)


    ax.set_title(title)

# -----------------------------------
# Example usage
sampling_rate = 2000
start_time = 3
end_time = 5

# Compute average powers
# average_power, _ = compute_average_power(extracted_segments, start_time, end_time, fs=sampling_rate)
# average_power_2, _ = compute_average_power(extracted_segments_ctrl, start_time, end_time, fs=sampling_rate)
# average_power, _ = compute_average_power(f_segments_delta1, start_time, end_time, fs=sampling_rate)
# average_power_2, _ = compute_average_power(f_segments_delta2, start_time, end_time, fs=sampling_rate)
average_power, _ = compute_average_power(f_segments_theta1, start_time, end_time, fs=sampling_rate)
average_power_2, _ = compute_average_power(f_segments_theta2, start_time, end_time, fs=sampling_rate)
# average_power, _ = compute_average_power(f_segments_gamma1, start_time, end_time, fs=sampling_rate)
# average_power_2, _ = compute_average_power(f_segments_gamma2, start_time, end_time, fs=sampling_rate)

""""Change the filtered_segments_number to which channels were stimulated during experiment"""
# Extract stimulation channels
ch1 = get_channel_from_name('filtered_segments_02')
ch2 = get_channel_from_name('filtered_segments_21')

# Define layout
layout = [
    ['A7', 'A6', 'A14', 'A20', 'B24', 'B25', 'B17', 'B11'],
    ['A0', 'A4', 'A13', 'A22', 'B31', 'B27', 'B18', 'B9'],
    ['A23', 'A17', 'A27', 'A11', 'B6', 'B30', 'B10', 'B20'],
    ['A31', 'A19', 'A12', 'A18', 'B0', 'B2', 'B19', 'B13'],
    ['A28', 'A2', 'A29', 'A15', 'B3', 'B29', 'B12', 'B16'],
    ['A5', 'A21', 'A1', 'A25', 'B26', 'B4', 'B14', 'B8'],
    ['A24', 'A3', 'A10', 'A16', 'B7', 'B28', 'B21', 'B15'],
    ['A26', 'A30', 'A8', 'A9', 'B5', 'B1', 'B23', 'B22']
]
"""MED64 channel layout"""
# layout = [
#             ['A22', 'B18', 'B30', 'B24', 'B1',  'B3',  'B5',  'B23'],
#             ['A23', 'B20', 'B16', 'B26', 'B25', 'B27', 'B22', 'A24'],
#             ['B21', 'B14', 'B15', 'B28', 'B2',  'B4',  'A0',  'A1' ],
#             ['B19', 'B12', 'B13', 'B11', 'B0',  'A25', 'A2',  'A3' ],
#             ['B10', 'B17', 'B9',  'A14', 'A26', 'A27', 'A5',  'A4' ],
#             ['B8',  'B31', 'A30', 'A20', 'A12', 'A29', 'A7',  'A6' ],
#             ['B7',  'B6',  'A16', 'A21', 'A13', 'A11', 'A31', 'A8' ],
#             ['B29', 'A28', 'A18', 'A15', 'A19', 'A17', 'A10', 'A9' ]
#         ] 
# Determine global vmin and vmax
vmin = min(average_power.min(), average_power_2.min())
vmax = max(average_power.max(), average_power_2.max())

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])  # for colorbar

plot_average_power(average_power, layout, selected_channel=ch1,
                   title='', vmin=vmin, vmax=vmax,
                   ax=ax0, show_colorbar=False)

plot_average_power(average_power_2, layout, selected_channel=ch2,
                   title='', vmin=vmin, vmax=vmax,
                   ax=ax1, show_colorbar=False)

# Add shared colorbar in designated axis
cbar = fig.colorbar(ax1.collections[0], cax=cax)
cbar.set_label('Average Power (log scale)', fontsize=25)
# Adjust the font size of the color bar ticks
cbar.ax.tick_params(labelsize=20)  # Change 16 to your desired font size
plt.tight_layout()
plt.show()


# In[] updated plot average power to filter out high impedance channels --- channel_names not defined
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
from scipy.signal import welch

def compute_average_power(extracted_segments, start_time, end_time, fs=2000):
    """
    Computes the average power for each channel across all trials in the extracted_segments array within a selected time window.

    Parameters:
    - extracted_segments: numpy array of shape (trials, channels, samples)
    - start_time: start time of the window in seconds
    - end_time: end time of the window in seconds
    - fs: sampling frequency (default: 2 kHz)

    Returns:
    - average_power: numpy array of average power for each channel (shape: channels,)
    """
    num_trials, num_channels, num_samples = extracted_segments.shape
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    average_power = np.zeros(num_channels)  # Store average power per channel

    for channel in range(num_channels):
        psd_list = []
        for trial in range(num_trials):
            f, Pxx = welch(extracted_segments[trial, channel, start_sample:end_sample], fs=fs)
            psd_list.append(Pxx)  # Store PSD for this trial

        # Compute mean power across trials and frequency bins
        # average_power[channel] = np.mean([np.mean(Pxx) for Pxx in psd_list])  
        average_power[channel] = np.mean(psd_list)  


    return average_power, psd_list


def get_channel_from_name(name):
    """
    Extracts the channel number from the filtered_segments variable name.

    Parameters:
    - name: string, the name of the filtered_segments variable (e.g., 'filtered_segments_25')

    Returns:
    - int: extracted channel number
    """
    # Extract the numeric part from the variable name (assumes the format 'filtered_segments_<channel>')
    return int(name.split('_')[-1])

def plot_average_power_filtered(average_power, layout, impedance_threshold=2000000,
                                title='Average Power Visualization', vmin=None, vmax=None):
    """
    Plots the average power of each channel within a specified layout, 
    filtering out channels with impedance above the threshold.

    Parameters:
    - average_power: numpy array of average power for each channel (shape: channels,)
    - layout: list of lists specifying the channel layout
    - impedance_dict: dictionary {channel_name: impedance_value}
    - impedance_threshold: maximum impedance allowed for inclusion (default: 2MΩ)
    - title: string, the title of the plot
    - vmin, vmax: normalization range for the heatmap color bar
    """
    num_rows = len(layout)
    num_cols = len(layout[0])
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10), sharex=True, sharey=True)
    # Convert to dictionary {channel_name: impedance}
    impedance_dict = dict(zip(channel_names, impedance_values))
    print("Impedance data loaded successfully.")
    
    for i, row in enumerate(layout):
        for j, channel_name in enumerate(row):
            ax = axes[i, j]

            try:
                # Get channel index and impedance value
                channel_idx = int(''.join(filter(str.isdigit, channel_name)))  
                impedance = impedance_dict.get(channel_name, float('inf'))  # Default to inf if missing

                # If impedance is too high, leave subplot blank
                if impedance > impedance_threshold:
                    ax.axis("off")
                else:
                    # Plot heatmap power for valid channels
                    ax.imshow([[average_power[channel_idx]]], cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
                    ax.set_title(channel_name, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

            except ValueError:
                ax.axis("off")  # Hide subplot if channel index extraction fails

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return impedance_dict
# Example usage:
# filtered_segments_25 = np.random.rand(10, 64, 6000)  # Example data for channel 25
# filtered_segments_27 = np.random.rand(10, 64, 6000)  # Example data for channel 27
sampling_rate = 2000  # Define the sampling rate (Hz)

# Compute average power for each set
average_power, psd_list = compute_average_power(f_segments_theta1, start_time=3, end_time=7, fs=sampling_rate)
average_power, psd_list = compute_average_power(f_segments_theta2, start_time=3, end_time=7, fs=sampling_rate)
# # Extract channel numbers from the names
ch1 = get_channel_from_name('filtered_segments_02')
ch2 = get_channel_from_name('filtered_segments_021')

# Find the global min and max for consistent normalization
vmin = average_power.min()
vmax = average_power.max()

# Define the channel layout as provided
layout = [
    ['A7', 'A6', 'A14', 'A20', 'B24', 'B25', 'B17', 'B11'],
    ['A0', 'A4', 'A13', 'A22', 'B31', 'B27', 'B18', 'B9'],
    ['A23', 'A17', 'A27', 'A11', 'B6', 'B30', 'B10', 'B20'],
    ['A31', 'A19', 'A12', 'A18', 'B0', 'B2', 'B19', 'B13'],
    ['A28', 'A2', 'A29', 'A15', 'B3', 'B29', 'B12', 'B16'],
    ['A5', 'A21', 'A1', 'A25', 'B26', 'B4', 'B14', 'B8'],
    ['A24', 'A3', 'A10', 'A16', 'B7', 'B28', 'B21', 'B15'],
    ['A26', 'A30', 'A8', 'A9', 'B5', 'B1', 'B23', 'B22']
]
# """MED64 channel layout"""
# layout = [
#             ['A22', 'B18', 'B30', 'B24', 'B1',  'B3',  'B5',  'B23'],
#             ['A23', 'B20', 'B16', 'B26', 'B25', 'B27', 'B22', 'A24'],
#             ['B21', 'B14', 'B15', 'B28', 'B2',  'B4',  'A0',  'A1' ],
#             ['B19', 'B12', 'B13', 'B11', 'B0',  'A25', 'A2',  'A3' ],
#             ['B10', 'B17', 'B9',  'A14', 'A26', 'A27', 'A5',  'A4' ],
#             ['B8',  'B31', 'A30', 'A20', 'A12', 'A29', 'A7',  'A6' ],
#             ['B7',  'B6',  'A16', 'A21', 'A13', 'A11', 'A31', 'A8' ],
#             ['B29', 'A28', 'A18', 'A15', 'A19', 'A17', 'A10', 'A9' ]
#         ] 
annotations = np.empty_like(layout, dtype=object)  # Create an array of the same shape as layout
# Plot the average power for each channel according to the layout with consistent color scale
start_time = 3
end_time = 7
impedance_dict = plot_average_power_filtered(average_power, layout, title='Average Power of time-series', vmin=vmin, vmax=vmax)
# plot_average_power_filtered(average_power_2, layout, selected_channel=ch2, title=f'Average Power - Stimulation at ch{ch2}', vmin=vmin, vmax=vmax)

# In[] updated plot average power to filter out high impedance channels with different method to mask them out
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
from scipy.signal import welch

def compute_average_power(extracted_segments, start_time, end_time, fs=2000):
    """
    Computes the average power for each channel across all trials in the extracted_segments array within a selected time window.

    Parameters:
    - extracted_segments: numpy array of shape (trials, channels, samples)
    - start_time: start time of the window in seconds
    - end_time: end time of the window in seconds
    - fs: sampling frequency (default: 2 kHz)

    Returns:
    - average_power: numpy array of average power for each channel (shape: channels,)
    """
    num_trials, num_channels, num_samples = extracted_segments.shape
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    average_power = np.zeros(num_channels)  # Store average power per channel

    for channel in range(num_channels):
        psd_list = []
        for trial in range(num_trials):
            f, Pxx = welch(extracted_segments[trial, channel, start_sample:end_sample], fs=fs)
            psd_list.append(Pxx)  # Store PSD for this trial

        # Compute mean power across trials and frequency bins
        # average_power[channel] = np.mean([np.mean(Pxx) for Pxx in psd_list])  
        average_power[channel] = np.mean(psd_list)  


    return average_power, psd_list


def get_channel_from_name(name):
    """
    Extracts the channel number from the filtered_segments variable name.

    Parameters:
    - name: string, the name of the filtered_segments variable (e.g., 'filtered_segments_25')

    Returns:
    - int: extracted channel number
    """
    # Extract the numeric part from the variable name (assumes the format 'filtered_segments_<channel>')
    return int(name.split('_')[-1])
def get_channel_name(mch):

    if mch<32:
        Channel=f'A{mch}'
    elif mch<32*2:
        Channel=f'B{mch-32}'
    elif mch<32*3:
        Channel=f'C{mch-32*2}'
    elif mch<32*4:
        Channel=f'D{mch-32*3}'
    return Channel
def plot_average_power_filtered(average_power, layout, impedance_threshold=2000000,
                                title='Average Power Visualization', vmin=None, vmax=None, interpolation='bicubic'):
    """
    Plots the average power of each channel within a specified layout, 
    filtering out channels with impedance above the threshold.

    Parameters:
    - average_power: numpy array of average power for each channel (shape: channels,)
    - layout: list of lists specifying the channel layout
    - impedance_threshold: maximum impedance allowed for inclusion (default: 2MΩ)
    - title: string, the title of the plot
    - vmin, vmax: normalization range for the heatmap color bar
    """
    num_rows = len(layout)
    num_cols = len(layout[0])
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16), sharex=True, sharey=True, dpi=300)

    print("Impedance data loaded successfully.")

    for mch in range(len(average_power)):  # Loop over channel indices
        Channel = get_channel_name(mch)  # Get corresponding channel name
        
        # Find the position of the channel in the layout
        row, col = next(
            ((r, c) for r, roww in enumerate(layout) for c, item in enumerate(roww) if item == Channel),
            (None, None)  # Default if not found
        )
        
        if row is None or col is None:
            continue  # Skip if the channel is not in the layout

        ax = axes[row, col]  # Get the corresponding subplot

        if float(impedance_info[:,4][mch]) > impedance_threshold:
            ax.axis('off')  # Turn off the axis for this subplot
            continue
        else:
            # Plot heatmap power for valid channels
            # fake_matrix = np.full((5, 5), average_power[mch])  # Expand to 5x5 identical values
            # im = ax.imshow(fake_matrix, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='bicubic')

            im = ax.imshow([[average_power[mch]]], cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
            # ax.set_title(Channel, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove ticks and axis values for other time-series plots
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)
            axes[row, col].spines['left'].set_visible(False)
            axes[row, col].spines['bottom'].set_visible(False)
    # plt.suptitle(title, fontsize=16)
    # Add the colorbar using the image object (im)
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.1, pad=0.01)

    # Adjust the layout
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title and colorbar
    plt.show()

# Example usage:
# filtered_segments_25 = np.random.rand(10, 64, 6000)  # Example data for channel 25
# filtered_segments_27 = np.random.rand(10, 64, 6000)  # Example data for channel 27
sampling_rate = 2000  # Define the sampling rate (Hz)

# Compute average power for each set
average_power, psd_list = compute_average_power(f_segments_theta1, start_time=3, end_time=5, fs=sampling_rate)
average_power, psd_list = compute_average_power(f_segments_theta2, start_time=3, end_time=5, fs=sampling_rate)

# # Extract channel numbers from the names
ch1 = get_channel_from_name('filtered_segments_02')
ch2 = get_channel_from_name('filtered_segments_021')

# Find the global min and max for consistent normalization
vmin = average_power.min()
vmax = average_power.max()

# Define the channel layout as provided
layout = [
    ['A7', 'A6', 'A14', 'A20', 'B24', 'B25', 'B17', 'B11'],
    ['A0', 'A4', 'A13', 'A22', 'B31', 'B27', 'B18', 'B9'],
    ['A23', 'A17', 'A27', 'A11', 'B6', 'B30', 'B10', 'B20'],
    ['A31', 'A19', 'A12', 'A18', 'B0', 'B2', 'B19', 'B13'],
    ['A28', 'A2', 'A29', 'A15', 'B3', 'B29', 'B12', 'B16'],
    ['A5', 'A21', 'A1', 'A25', 'B26', 'B4', 'B14', 'B8'],
    ['A24', 'A3', 'A10', 'A16', 'B7', 'B28', 'B21', 'B15'],
    ['A26', 'A30', 'A8', 'A9', 'B5', 'B1', 'B23', 'B22']
]
# """MED64 channel layout"""
# layout = [
#             ['A22', 'B18', 'B30', 'B24', 'B1',  'B3',  'B5',  'B23'],
#             ['A23', 'B20', 'B16', 'B26', 'B25', 'B27', 'B22', 'A24'],
#             ['B21', 'B14', 'B15', 'B28', 'B2',  'B4',  'A0',  'A1' ],
#             ['B19', 'B12', 'B13', 'B11', 'B0',  'A25', 'A2',  'A3' ],
#             ['B10', 'B17', 'B9',  'A14', 'A26', 'A27', 'A5',  'A4' ],
#             ['B8',  'B31', 'A30', 'A20', 'A12', 'A29', 'A7',  'A6' ],
#             ['B7',  'B6',  'A16', 'A21', 'A13', 'A11', 'A31', 'A8' ],
#             ['B29', 'A28', 'A18', 'A15', 'A19', 'A17', 'A10', 'A9' ]
#         ] 
annotations = np.empty_like(layout, dtype=object)  # Create an array of the same shape as layout
# Plot the average power for each channel according to the layout with consistent color scale
start_time = 3
end_time = 5
plot_average_power_filtered(average_power, layout, title='Average Power of time-series', vmin=vmin, vmax=vmax, interpolation='bicubic')
plot_average_power_filtered(average_power_2, layout, title=f'Average Power - Stimulation at ch{ch2}', vmin=vmin, vmax=vmax)

# In[] # space for colorbar and red outline border of stimulated channel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
from scipy.signal import welch

def compute_average_power(extracted_segments, start_time, end_time, fs=2000):
    """
    Computes the average power for each channel across all trials in the extracted_segments array within a selected time window.

    Parameters:
    - extracted_segments: numpy array of shape (trials, channels, samples)
    - start_time: start time of the window in seconds
    - end_time: end time of the window in seconds
    - fs: sampling frequency (default: 2 kHz)

    Returns:
    - average_power: numpy array of average power for each channel (shape: channels,)
    """
    num_trials, num_channels, num_samples = extracted_segments.shape
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    average_power = np.zeros(num_channels)  # Store average power per channel

    for channel in range(num_channels):
        psd_list = []
        for trial in range(num_trials):
            f, Pxx = welch(extracted_segments[trial, channel, start_sample:end_sample], fs=fs)
            psd_list.append(Pxx)  # Store PSD for this trial

        # Compute mean power across trials and frequency bins
        # average_power[channel] = np.mean([np.mean(Pxx) for Pxx in psd_list])  
        average_power[channel] = np.mean(psd_list)  


    return average_power, psd_list


def get_channel_from_name(name):
    """
    Extracts the channel number from the filtered_segments variable name.

    Parameters:
    - name: string, the name of the filtered_segments variable (e.g., 'filtered_segments_25')

    Returns:
    - int: extracted channel number
    """
    # Extract the numeric part from the variable name (assumes the format 'filtered_segments_<channel>')
    return int(name.split('_')[-1])
def get_channel_name(mch):

    if mch<32:
        Channel=f'A{mch}'
    elif mch<32*2:
        Channel=f'B{mch-32}'
    elif mch<32*3:
        Channel=f'C{mch-32*2}'
    elif mch<32*4:
        Channel=f'D{mch-32*3}'
    return Channel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

def plot_average_power_filtered(average_power, layout, stim_channel=None, impedance_threshold=2000000,
                                title='Average Power Visualization', vmin=None, vmax=None):
    """
    Plots the average power of each channel within a specified layout,
    filtering out channels with impedance above the threshold.
    Highlights the stimulation channel with a red border.

    Parameters:
    - average_power: numpy array of average power for each channel (shape: channels,)
    - layout: list of lists specifying the channel layout
    - stim_channel: str, the name of the stimulation channel (e.g., 'A5')
    - impedance_threshold: maximum impedance allowed for inclusion (default: 2MΩ)
    - title: string, the title of the plot
    - vmin, vmax: normalization range for the heatmap color bar
    """
    num_rows = len(layout)
    num_cols = len(layout[0])
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16), sharex=True, sharey=True, dpi=300)

    for mch in range(len(average_power)):  # Loop over channel indices
        Channel = get_channel_name(mch)  # Get corresponding channel name
        
        # Find the position of the channel in the layout
        row, col = next(
            ((r, c) for r, roww in enumerate(layout) for c, item in enumerate(roww) if item == Channel),
            (None, None)  # Default if not found
        )
        
        if row is None or col is None:
            continue  # Skip if the channel is not in the layout

        ax = axes[row, col]  # Get the corresponding subplot

        # Check impedance threshold and skip if too high
        if float(impedance_info[:,4][mch]) > impedance_threshold:
            ax.axis('off')  # Hide subplot
            continue

        # Plot heatmap power for valid channels
        im = ax.imshow([[average_power[mch]]], cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))

        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add a red border if this is the stimulation channel
        if Channel == stim_channel:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                   fill=False, edgecolor='red', linewidth=10))

    # Add the colorbar using the last valid image object
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.1, pad=0.01)

    # plt.suptitle(title, fontsize=16)
    plt.show()

# Example usage:

layout = [
    ['A7', 'A6', 'A14', 'A20', 'B24', 'B25', 'B17', 'B11'],
    ['A0', 'A4', 'A13', 'A22', 'B31', 'B27', 'B18', 'B9'],
    ['A23', 'A17', 'A27', 'A11', 'B6', 'B30', 'B10', 'B20'],
    ['A31', 'A19', 'A12', 'A18', 'B0', 'B2', 'B19', 'B13'],
    ['A28', 'A2', 'A29', 'A15', 'B3', 'B29', 'B12', 'B16'],
    ['A5', 'A21', 'A1', 'A25', 'B26', 'B4', 'B14', 'B8'],
    ['A24', 'A3', 'A10', 'A16', 'B7', 'B28', 'B21', 'B15'],
    ['A26', 'A30', 'A8', 'A9', 'B5', 'B1', 'B23', 'B22']
]

stim_channel1 = 'A2'  # Set the stimulation channel
stim_channel2 = 'A21'
# Compute average power for each set
# Plot the average power for each channel according to the layout with consistent color scale
start_time = 3
end_time = 5
average_power, psd_list = compute_average_power(f_segments_theta1, start_time=3, end_time=5, fs=sampling_rate)
average_power_2, psd_list_2 = compute_average_power(f_segments_theta2, start_time=3, end_time=5, fs=sampling_rate)

# Find the global min and max for consistent normalization
vmin = min(average_power.min(), average_power_2.min())
vmax = max(average_power.max(), average_power_2.max())

plot_average_power_filtered(average_power, layout, stim_channel=stim_channel1, vmin=vmin, vmax=vmax)
plot_average_power_filtered(average_power_2, layout, stim_channel=stim_channel2, vmin=vmin, vmax=vmax)

# In[] delta

start_time = 2
end_time = 5
average_power, psd_list = compute_average_power(f_segments_delta1, start_time=3, end_time=5, fs=sampling_rate)
average_power_2, psd_list_2 = compute_average_power(f_segments_delta2, start_time=3, end_time=5, fs=sampling_rate)

# Find the global min and max for consistent normalization
vmin = min(average_power.min(), average_power_2.min())
vmax = max(average_power.max(), average_power_2.max())

# Plot
plot_average_power_filtered(average_power, layout, stim_channel=stim_channel1, vmin=vmin, vmax=vmax)
plot_average_power_filtered(average_power_2, layout, stim_channel=stim_channel2, vmin=vmin, vmax=vmax)

# In[]  trial to trial variability for psd

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_trial_psd(extracted_segments, channel, start_time, end_time, fs=2000, nperseg=256):
    """
    Plots the power spectral density (PSD) for each trial in a given channel.

    Parameters:
    - extracted_segments: numpy array of shape (trials, channels, samples)
    - channel: int, the index of the channel to analyze
    - start_time: float, start time of the window in seconds
    - end_time: float, end time of the window in seconds
    - fs: int, sampling frequency (default: 2000 Hz)
    - nperseg: int, segment length for Welch’s method (default: 256)
    """
    num_trials, num_channels, num_samples = extracted_segments.shape
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    plt.figure(figsize=(10, 6))

    for trial in range(num_trials):
        # Compute PSD using Welch's method
        f, Pxx = welch(extracted_segments[trial, channel, start_sample:end_sample], fs=fs, nperseg=nperseg)
        
        # Plot PSD for each trial
        plt.semilogy(f, Pxx, alpha=0.5)  # Use semi-log plot for better visualization

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (µV²/Hz)")
    plt.xlim(0,300)
    plt.title(f"Trial-wise PSD for Channel {channel}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Example usage:
plot_trial_psd(extracted_segments, channel=9, start_time=3, end_time=7, fs=2000)

# In[] trial-trial spectrograms of particular channel
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_trial_spectrogram(extracted_segments, channel, start_time, end_time, fs=2000, nperseg=512):
    """
    Plots the spectrogram for each trial in a given channel.

    Parameters:
    - extracted_segments: numpy array of shape (trials, channels, samples)
    - channel: int, the index of the channel to analyze
    - start_time: float, start time of the window in seconds
    - end_time: float, end time of the window in seconds
    - fs: int, sampling frequency (default: 2000 Hz)
    - nperseg: int, segment length for the spectrogram
    """
    num_trials, num_channels, num_samples = extracted_segments.shape
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    plt.figure(figsize=(10, 6))

    for trial in range(num_trials):
        # Extract the signal for the current trial and channel
        signal = extracted_segments[trial, channel, start_sample:end_sample]

        # Compute the spectrogram using scipy's spectrogram function
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)

        # Plot the spectrogram of each trial
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', alpha=1)  # dB scale

        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0,100)
        plt.axvline(x=stim_time, color='red',
                    linestyle='--', linewidth=3, label='STIM')
        plt.title(f"Trial {trial+1} Spectrogram for Channel {channel}")
        plt.colorbar(label="Power (dB)")
        plt.legend()
        # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

# Example usage:
stim_time = 2
plot_trial_spectrogram(extracted_segments, channel=13, start_time=0, end_time=7, fs=2000)

# In[] Plot for figure 6 c and d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_selected_trials_spectrogram(extracted_segments, trials_to_plot=[0, 1, 2, 3, 4], ch_idx=13, fs=2000, nperseg=500, noverlap=250):
    """
    Plots spectrograms for selected trials from the extracted segments array.

    Parameters:
    - extracted_segments: numpy array of shape (trials, channels, samples)
    - trials_to_plot: list of trial indices to plot (default: [0, 1, 2, 3, 4])
    - fs: sampling frequency (default: 2000 Hz)
    - nperseg: length of each segment for the spectrogram
    - noverlap: number of points to overlap between segments
    """
    num_trials, num_channels, num_samples = extracted_segments.shape

    # Check that the trials_to_plot are within the range of available trials
    trials_to_plot = [trial for trial in trials_to_plot if trial < num_trials]
    num_plots = len(trials_to_plot)

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 12))

    for i, trial_idx in enumerate(trials_to_plot):
        # For each selected trial, plot the spectrogram
        f, t, Sxx = spectrogram(extracted_segments[trial_idx, ch_idx], fs=fs, nperseg=nperseg, noverlap=noverlap)

        # Plot the spectrogram on the ith subplot
        im = axes[i].pcolormesh(t, f, np.log10(Sxx), shading='auto')  # Log scale for better visualization
        # axes[i].set_ylabel(f'Trial {trial_idx + 1}', rotation=0)  # Set y label horizontally
        axes[i].set_ylim(0, 100)  # Set frequency range
        axes[i].axvline(x=stim_time, color='red',
                    linestyle='--', linewidth=3, label='STIM')
        # Set ticks only for the 5th subplot (i == 4)
        if i == 4:
            axes[i].tick_params(axis='both', which='both', direction='in', length=6)
        else:
            axes[i].tick_params(axis='both', which='both', length=0)  # Remove ticks for other subplots
            axes[i].set_xticks([])  # Remove x-axis ticks
            axes[i].set_yticks([])  # Remove y-axis ticks
    # Add colorbar for the spectrograms
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.1, pad=0.05)
    
    # Set global title for the entire figure
    # fig.suptitle('Trial Variability', fontsize=30)
    fig.supxlabel('Seconds')
    fig.supylabel('Frequency(Hz)')
    
    # plt.tight_layout()
    plt.show()

# Example usage:
# Assuming `extracted_segments` is your data array with shape (trials, channels, samples)
stim_time = 2
ch_idx = 13
plot_selected_trials_spectrogram(extracted_segments, trials_to_plot=[4, 17, 23, 35, 48], ch_idx=ch_idx, fs=2000)

# In[]
def compare_theta_power_over_trials(f_segments1, f_segments2, channel_idx, fs=2000, start_sec=3, end_sec=7):
    """
    Computes and compares theta band power across trials for two datasets.

    Parameters:
        f_segments1, f_segments2 (ndarray): shape [trials, channels, timepoints]
        channel_idx (int): channel to analyze
        fs (int): sampling frequency
        start_sec, end_sec (float): time window in seconds

    Returns:
        power1, power2: power values per trial for both datasets
    """
    def compute_power(f_segments):
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        return np.array([
            np.mean(f_segments[trial, channel_idx, start_idx:end_idx] ** 2)
            for trial in range(f_segments.shape[0])
        ])

    power1 = compute_power(f_segments1)
    power2 = compute_power(f_segments2)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(power1) + 1), power1, label="Condition 1", marker='o')
    plt.plot(range(1, len(power2) + 1), power2, label="Condition 2", marker='s')
    plt.title(f"Theta Power (Channel {channel_idx}) from {start_sec}s to {end_sec}s")
    plt.xlabel("Trial")
    plt.ylabel("Theta Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return power1, power2
power_theta1, power_theta2 = compare_theta_power_over_trials(f_segments_theta1, f_segments_theta2, channel_idx=30)

# In[] plot theta values over trials
def compare_theta_power_all_channels(f_segments1, f_segments2, fs=2000, start_sec=3, end_sec=4):
    """
    Computes and compares gamma power per trial for all channels and plots each channel on a separate graph.

    Parameters:
        f_segments1, f_segments2 (ndarray): shape [trials, channels, timepoints]
        fs (int): sampling frequency
        start_sec, end_sec (float): time window in seconds

    Returns:
        powers1, powers2: lists of arrays containing power per trial for each channel
    """
    num_channels = f_segments1.shape[1]
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    powers1 = []
    powers2 = []

    for ch in range(num_channels):
        power1 = np.array([
            np.mean(f_segments1[trial, ch, start_idx:end_idx] ** 2)
            for trial in range(f_segments1.shape[0])
        ])
        power2 = np.array([
            np.mean(f_segments2[trial, ch, start_idx:end_idx] ** 2)
            for trial in range(f_segments2.shape[0])
        ])

        powers1.append(power1)
        powers2.append(power2)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(power1) + 1), power1, label="Input A", marker='o')
        plt.plot(range(1, len(power2) + 1), power2, label="Input B", marker='s')
        plt.title(f"Theta Power Over Trials - Channel {ch} ({start_sec}s to {end_sec}s)")
        plt.xlabel("Trial")
        plt.ylabel("Theta Power")
        plt.ylim(0,1500)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return powers1, powers2

start_sec = 3
end_sec = 7
power_all_theta1, power_all_theta2 = compare_theta_power_all_channels(
    f_segments_theta1, f_segments_theta2, fs=2000, start_sec=start_sec, end_sec=end_sec
)
# In[]
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_theta_power_bar_with_error(f_segments1, f_segments2, fs=2000, start_sec=3, end_sec=4):
    """
    Computes average gamma power per channel across all trials and compares
    the two conditions using a bar graph with error bars.

    Parameters:
        f_segments1, f_segments2 (ndarray): shape [trials, channels, timepoints]
        fs (int): sampling frequency
        start_sec, end_sec (float): time window in seconds
    """
    num_channels = f_segments1.shape[1]
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    mean_power1, sem_power1 = [], []
    mean_power2, sem_power2 = [], []

    for ch in range(num_channels):
        power1 = np.mean(f_segments1[:, ch, start_idx:end_idx] ** 2, axis=1)
        power2 = np.mean(f_segments2[:, ch, start_idx:end_idx] ** 2, axis=1)

        mean_power1.append(np.mean(power1))
        sem_power1.append(np.std(power1, ddof=1) / np.sqrt(len(power1)))

        mean_power2.append(np.mean(power2))
        sem_power2.append(np.std(power2, ddof=1) / np.sqrt(len(power2)))

    x = np.arange(num_channels)
    width = 0.35

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, mean_power1, width, yerr=sem_power1, capsize=5, label='Condition 1', alpha=0.8)
    plt.bar(x + width/2, mean_power2, width, yerr=sem_power2, capsize=5, label='Condition 2', alpha=0.8)

    plt.xlabel('Channel', fontsize=14)
    plt.ylabel('Mean Theta Power ± SEM', fontsize=14)
    plt.title(f'Mean Theta Power per Channel ({start_sec}s to {end_sec}s)', fontsize=16)
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
plot_avg_theta_power_bar_with_error(f_segments_theta1, f_segments_theta2, fs=2000, start_sec=2, end_sec=4)

# In[] plot gamma values over trials
def compare_gamma_power_all_channels(f_segments1, f_segments2, fs=2000, start_sec=3, end_sec=4):
    """
    Computes and compares gamma power per trial for all channels and plots each channel on a separate graph.

    Parameters:
        f_segments1, f_segments2 (ndarray): shape [trials, channels, timepoints]
        fs (int): sampling frequency
        start_sec, end_sec (float): time window in seconds

    Returns:
        powers1, powers2: lists of arrays containing power per trial for each channel
    """
    num_channels = f_segments1.shape[1]
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    powers1 = []
    powers2 = []

    for ch in range(num_channels):
        power1 = np.array([
            np.mean(f_segments1[trial, ch, start_idx:end_idx] ** 2)
            for trial in range(f_segments1.shape[0])
        ])
        power2 = np.array([
            np.mean(f_segments2[trial, ch, start_idx:end_idx] ** 2)
            for trial in range(f_segments2.shape[0])
        ])

        powers1.append(power1)
        powers2.append(power2)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(power1) + 1), power1, label="Input A", marker='o')
        plt.plot(range(1, len(power2) + 1), power2, label="Input B", marker='s')
        plt.title(f"Gamma Power Over Trials - Channel {ch} ({start_sec}s to {end_sec}s)")
        plt.xlabel("Trial")
        plt.ylabel("Gamma Power")
        plt.ylim(0,35000)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return powers1, powers2

start_sec = 2
end_sec = 2.5
power_all_gamma1, power_all_gamma2 = compare_gamma_power_all_channels(
    f_segments_gamma1, f_segments_gamma2, fs=2000, start_sec=start_sec, end_sec=end_sec
)
# In[]

# In[] all channels on one bar graph
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_gamma_power_bar_with_error(f_segments1, f_segments2, fs=2000, start_sec=3, end_sec=4):
    """
    Computes average gamma power per channel across all trials and compares
    the two conditions using a bar graph with error bars.

    Parameters:
        f_segments1, f_segments2 (ndarray): shape [trials, channels, timepoints]
        fs (int): sampling frequency
        start_sec, end_sec (float): time window in seconds
    """
    num_channels = f_segments1.shape[1]
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    mean_power1, sem_power1 = [], []
    mean_power2, sem_power2 = [], []

    for ch in range(num_channels):
        power1 = np.mean(f_segments1[:, ch, start_idx:end_idx] ** 2, axis=1)
        power2 = np.mean(f_segments2[:, ch, start_idx:end_idx] ** 2, axis=1)

        mean_power1.append(np.mean(power1))
        sem_power1.append(np.std(power1, ddof=1) / np.sqrt(len(power1)))

        mean_power2.append(np.mean(power2))
        sem_power2.append(np.std(power2, ddof=1) / np.sqrt(len(power2)))

    x = np.arange(num_channels)
    width = 0.35

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, mean_power1, width, yerr=sem_power1, capsize=5, label='Condition 1', alpha=0.8)
    plt.bar(x + width/2, mean_power2, width, yerr=sem_power2, capsize=5, label='Condition 2', alpha=0.8)

    plt.xlabel('Channel', fontsize=14)
    plt.ylabel('Mean Gamma Power ± SEM', fontsize=14)
    plt.title(f'Mean Gamma Power per Channel ({start_sec}s to {end_sec}s)', fontsize=16)
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
plot_avg_gamma_power_bar_with_error(f_segments_gamma1, f_segments_gamma2, fs=2000, start_sec=2, end_sec=2.5)
# In[] plot bar graph for selected channels of gamma power
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_power_selected_channels(f_segments1, f_segments2, selected_channels, fs=2000, start_sec=3, end_sec=4):
    """
    Plots individual bar graphs comparing mean power between two conditions
    for selected channels only, with SEM error bars.

    Parameters:
        f_segments1, f_segments2: np.ndarray
            Arrays of shape [trials, channels, timepoints]
        selected_channels: list of int
            Indices of channels to plot
        fs: int
            Sampling frequency
        start_sec, end_sec: float
            Time window for computing power
    """
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    for ch in selected_channels:
        power1 = np.mean(f_segments1[:, ch, start_idx:end_idx] ** 2, axis=1)
        power2 = np.mean(f_segments2[:, ch, start_idx:end_idx] ** 2, axis=1)

        mean1 = np.mean(power1)
        mean2 = np.mean(power2)
        sem1 = np.std(power1, ddof=1) / np.sqrt(len(power1))
        sem2 = np.std(power2, ddof=1) / np.sqrt(len(power2))

        # Plotting per selected channel
        plt.figure(figsize=(6, 4))
        plt.bar(['Input A', 'Input B'], [mean1, mean2],
                yerr=[sem1, sem2], capsize=6, color=['skyblue', 'salmon'])

        plt.title(f'Channel {ch} - Power Comparison', fontsize=14)
        plt.ylabel('Mean Power ± SEM', fontsize=12)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()
# To plot only channels 3, 5, and 12:
plot_avg_power_selected_channels(f_segments_gamma1, f_segments_gamma2, selected_channels=[13, 14, 15], start_sec=3, end_sec=3.5)
plot_avg_power_selected_channels(f_segments_theta1, f_segments_theta2, selected_channels=[13, 14, 15], start_sec=3, end_sec=7)
# In[]
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_power_selected_channels_together(f_segments1, f_segments2, selected_channels, fs=2000, start_sec=3, end_sec=4):
    """
    Plots a grouped bar graph comparing mean power between two conditions
    for selected channels.

    Parameters:
        f_segments1, f_segments2: np.ndarray
            Arrays of shape [trials, channels, timepoints]
        selected_channels: list of int
            Indices of channels to plot
        fs: int
            Sampling frequency
        start_sec, end_sec: float
            Time window for computing power
    """
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    mean_powers1 = []
    mean_powers2 = []
    sem_powers1 = []
    sem_powers2 = []

    for ch in selected_channels:
        power1 = np.mean(f_segments1[:, ch, start_idx:end_idx] ** 2, axis=1)
        power2 = np.mean(f_segments2[:, ch, start_idx:end_idx] ** 2, axis=1)

        mean_powers1.append(np.mean(power1))
        mean_powers2.append(np.mean(power2))
        sem_powers1.append(np.std(power1, ddof=1) / np.sqrt(len(power1)))
        sem_powers2.append(np.std(power2, ddof=1) / np.sqrt(len(power2)))

    # Plotting
    x = np.arange(len(selected_channels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, mean_powers1, yerr=sem_powers1, width=width, label='Input A',
            color='skyblue', capsize=5)
    plt.bar(x + width/2, mean_powers2, yerr=sem_powers2, width=width, label='Input B',
            color='salmon', capsize=5)

    plt.xticks(x, [f'Ch {ch}' for ch in selected_channels], rotation=45)
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("Mean Power ± SEM", fontsize=12)
    plt.title(f"Mean Power Comparison ({start_sec}s–{end_sec}s)", fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
plot_avg_power_selected_channels_together(f_segments_gamma1, f_segments_gamma2, selected_channels=[0, 1, 2, 4, 5, 6, 7, 13, 14, 15], start_sec=3, end_sec=3.5)
plot_avg_power_selected_channels_together(f_segments_theta1, f_segments_theta2, selected_channels=[0, 1, 2, 4, 5, 6, 7, 13, 14, 15], start_sec=2, end_sec=3)

# In[]
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def plot_avg_power_selected_channels_together_with_stats(f_segments1, f_segments2, selected_channels, fs=2000, start_sec=3, end_sec=4):
    """
    Plots grouped bar graph comparing mean power with statistical significance (Welch's t-test)
    for selected channels across two conditions.

    Parameters:
        f_segments1, f_segments2: np.ndarray
            Arrays of shape [trials, channels, timepoints]
        selected_channels: list of int
            Indices of channels to plot
        fs: int
            Sampling frequency
        start_sec, end_sec: float
            Time window for computing power
    """
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    mean_powers1 = []
    mean_powers2 = []
    sem_powers1 = []
    sem_powers2 = []
    p_values = []

    for ch in selected_channels:
        power1 = np.mean(f_segments1[:, ch, start_idx:end_idx] ** 2, axis=1)
        power2 = np.mean(f_segments2[:, ch, start_idx:end_idx] ** 2, axis=1)

        mean_powers1.append(np.mean(power1))
        mean_powers2.append(np.mean(power2))
        sem_powers1.append(np.std(power1, ddof=1) / np.sqrt(len(power1)))
        sem_powers2.append(np.std(power2, ddof=1) / np.sqrt(len(power2)))

        # Welch’s t-test
        _, p = ttest_ind(power1, power2, equal_var=False)
        p_values.append(p)

    # Plotting
    x = np.arange(len(selected_channels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, mean_powers1, yerr=sem_powers1, width=width,
                    label='Input A', color='skyblue', capsize=5)
    bars2 = plt.bar(x + width/2, mean_powers2, yerr=sem_powers2, width=width,
                    label='Input B', color='salmon', capsize=5)

    # Add significance annotations
    for i, p in enumerate(p_values):
        # Determine significance label
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        # Get y-position above bars
        y = max(mean_powers1[i] + sem_powers1[i], mean_powers2[i] + sem_powers2[i])
        plt.text(x[i], y + 0.02 * y, sig, ha='center', va='bottom', fontsize=12)

    plt.xticks(x, [f'Ch {ch}' for ch in selected_channels], rotation=45)
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("Mean Power ± SEM", fontsize=12)
    plt.title(f"Mean Power Comparison ({start_sec}s–{end_sec}s) with Significance", fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_avg_power_selected_channels_together_with_stats(
#     f_segments_gamma1, f_segments_gamma2,
#     selected_channels=[13, 14, 15],
#     start_sec=3, end_sec=3.5
# )
plot_avg_power_selected_channels_together_with_stats(
    f_segments_delta1, f_segments_delta2,
    selected_channels=[0, 1, 2, 4, 5, 6, 7, 13, 14, 15],
    start_sec=2, end_sec=7
)
plot_avg_power_selected_channels_together_with_stats(
    f_segments_theta1, f_segments_theta2,
    selected_channels=[0, 1, 2, 4, 5, 6, 7, 13, 14, 15],
    start_sec=2, end_sec=7
)

plot_avg_power_selected_channels_together_with_stats(
    f_segments_gamma1, f_segments_gamma2,
    selected_channels=[0, 1, 2, 4, 5, 6, 7, 13, 14, 15],
    start_sec=2, end_sec=7
)
# # In[]
# def plot_trial_variance_per_channel(f_segments1, f_segments2, fs=2000, start_sec=3, end_sec=7):
#     """
#     Plots trial-wise variance for each channel from two filtered datasets.
    
#     Parameters:
#         f_segments1, f_segments2: np.array of shape [trials, channels, timepoints]
#         fs: Sampling frequency
#         start_sec, end_sec: Time window in seconds for variance calculation
#     """
#     start_idx = int(start_sec * fs)
#     end_idx = int(end_sec * fs)
#     n_channels = f_segments1.shape[1]
    
#     for ch in range(n_channels):
#         var1 = np.var(f_segments1[:, ch, start_idx:end_idx], axis=1)
#         var2 = np.var(f_segments2[:, ch, start_idx:end_idx], axis=1)

#         plt.figure(figsize=(8, 4))
#         plt.plot(var1, label="Condition 1", marker='o')
#         plt.plot(var2, label="Condition 2", marker='s')
#         plt.title(f"Channel {ch} - Trial Variance ({start_sec}-{end_sec}s)")
#         plt.xlabel("Trial")
#         plt.ylabel("Variance")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
# plot_trial_variance_per_channel(f_segments_theta1, f_segments_theta2)
