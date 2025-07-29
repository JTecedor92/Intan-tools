import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter.simpledialog import askinteger
from tkinter import filedialog
from scipy.signal import spectrogram

from intanutil.data import (
    calculate_data_size,
    read_all_data_blocks,
    check_end_of_file,
    parse_data,
    data_to_result
)
from intanutil.filter import apply_notch_filter
from intanutil.header import read_header

# --- Select RHS file using file dialog ---
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select RHS file",
    filetypes=[("RHS files", "*.rhs")],
    initialdir="C:/Internship Data"
)

# Change initialdir to valid file path

if not file_path:
    print("No file selected.")
    exit()

print(f"\nSelected file: {file_path}")

# --- Open file and read header ---
with open(file_path, 'rb') as fid:
    header = read_header(fid)
    data_present, filesize, num_blocks, num_samples = calculate_data_size(header, file_path, fid)
    data = read_all_data_blocks(header, num_samples, num_blocks, fid)
    check_end_of_file(filesize, fid)

# --- Post-processing ---
apply_notch_filter(header, data)
parse_data(header, data)
result = data_to_result(header, data, {})

amplifier_data = result['amplifier_data']
sample_rate = header['sample_rate']

# --- Channel Selection for Spectrogram ---``
channel_idx = askinteger("Select Channel", "Enter channel index (0-63):", minvalue=0, maxvalue=63)
if channel_idx is None:
    print("Channel selection canceled.")
    exit()
signal = amplifier_data[channel_idx]

# --- Animation Parameters ---
window_size = 20000        # samples per window
step_size = 250            # hop size
n_samples = len(signal)
n_frames = (n_samples - window_size) // step_size

# --- Spectrogram Computation Function ---
def compute_spectrogram(signal_window, fs):
    f, t, Sxx = spectrogram(signal_window, fs=fs, nperseg=window_size//2, noverlap=window_size//4)
    Sxx[Sxx == 0] = 1e-12  # Avoid log(0)
    return f, t, Sxx

# --- Setup Figure for Spectrogram Animation ---
fig, ax = plt.subplots()
f_init, t_init, Sxx_init = compute_spectrogram(signal[:window_size], sample_rate)

img = ax.imshow(10 * np.log10(Sxx_init), aspect='auto', origin='lower',
                extent=[t_init[0], t_init[-1], f_init[0], f_init[-1]],
                cmap='viridis')
cb = plt.colorbar(img, ax=ax)
cb.set_label("Power (dB)")
ax.set_ylim(0,200)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")

# --- Animation Update Function ---
def update_spectrogram(frame_idx):
    start = frame_idx * step_size
    end = start + window_size
    if end > n_samples:
        return []

    signal_window = signal[start:end]
    f, t, Sxx = compute_spectrogram(signal_window, sample_rate)
    img.set_data(10 * np.log10(Sxx))
    img.set_extent([start / sample_rate, (start + window_size) / sample_rate, f[0], f[-1]])
    ax.set_xlim(start / sample_rate, (start + window_size) / sample_rate)
    ax.set_title(f"Spectrogram (Channel {channel_idx}) – {start/sample_rate:.2f}s to {end/sample_rate:.2f}s")
    return [img]

def update_save(frame_idx):
    start = frame_idx * step_size
    end = start + window_size
    if end > n_samples:
        return []

    signal_window = signal[start:end]
    f, t, Sxx = compute_spectrogram(signal_window, sample_rate)
    img_save.set_data(10 * np.log10(Sxx))
    img_save.set_extent([start / sample_rate, (start + window_size) / sample_rate, f[0], f[-1]])
    ax_save.set_xlim(start / sample_rate, (start + window_size) / sample_rate)
    ax_save.set_title(f"Spectrogram (Channel {channel_idx}) – {start/sample_rate:.2f}s to {end/sample_rate:.2f}s")
    return [img_save]

# === FULL ANIMATION FOR DISPLAY ===
ani = animation.FuncAnimation(fig, update_spectrogram, frames=n_frames, blit=False, interval=50)

# === LIMITED ANIMATION FOR SAVING ===
# Create a hidden figure (so it doesn't interfere)
fig_save, ax_save = plt.subplots()
f_init, t_init, Sxx_init = compute_spectrogram(signal[:window_size], sample_rate)

img_save = ax_save.imshow(10 * np.log10(Sxx_init), aspect='auto', origin='lower',
                extent=[t_init[0], t_init[-1], f_init[0], f_init[-1]],
                cmap='viridis')
cb_save = plt.colorbar(img_save, ax=ax_save)
cb_save.set_label("Power (dB)")
ax_save.set_ylim(0,200)
ax_save.set_xlabel("Time (s)")
ax_save.set_ylabel("Frequency (Hz)")

# 10 seconds × 20 fps = 200 frames
frames_to_save = min(200, n_frames)
ani_save = animation.FuncAnimation(fig_save, update_save, frames=frames_to_save, blit=False, interval=50)

# Save to GIF
save_path = filedialog.asksaveasfilename(
    title="Save GIF as",
    defaultextension=".gif",
    filetypes=[("GIF files", "*.gif")],
    initialdir="C:\Internship Data"
)

# Change initialdir to valid file path

if save_path:
    ani_save.save(save_path, writer='pillow', fps=20)
    print(f"GIF saved to: {save_path}")
else:
    print("GIF save canceled.")

# Close the hidden figure (optional cleanup)
plt.close(fig_save)

# === DISPLAY FULL ANIMATION ===
plt.show()