import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter.simpledialog import askinteger
from tkinter import filedialog
from scipy.signal import butter, sosfiltfilt

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
    # Header must be parsed according to your custom logic
    header = read_header(fid)

    data_present, filesize, num_blocks, num_samples = calculate_data_size(header, file_path, fid)
    data = read_all_data_blocks(header, num_samples, num_blocks, fid)
    check_end_of_file(filesize, fid)

# --- Post-processing ---
def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0 or high >= 1 or low >= high:
        raise ValueError(f"Invalid bandpass range: lowcut {lowcut} Hz, highcut {highcut} Hz")

    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, data, axis=1)

apply_notch_filter(header, data)
parse_data(header, data)
result = data_to_result(header, data, {})

amplifier_data = result['amplifier_data'][:64]
sample_rate = header['sample_rate']

# Define filter bounds
lowcut = askinteger("Select Channel", "Enter filter lower bound (Hz):", minvalue=0.00001)
highcut = askinteger("Select Channel", "Enter filter upper bound (Hz):", minvalue= lowcut + 0.00001)
if lowcut is None or highcut is None:
    print("Filter specifications not fulfilled.")
    exit()

# Apply band-pass filter
amplifier_data = apply_bandpass_filter(amplifier_data, lowcut, highcut, sample_rate)
print("Filtered amplifier data range:", np.min(amplifier_data), "to", np.max(amplifier_data))

# --- Custom 8x8 Layout ---
layout_names = [
    ['A-007', 'A-006', 'A-014', 'A-020', 'B-024', 'B-025', 'B-017', 'B-011'],
    ['A-000', 'A-004', 'A-013', 'A-022', 'B-031', 'B-027', 'B-018', 'B-009'],
    ['A-023', 'A-017', 'A-027', 'A-011', 'B-006', 'B-030', 'B-010', 'B-020'],
    ['A-031', 'A-019', 'A-012', 'A-018', 'B-000', 'B-002', 'B-019', 'B-013'],
    ['A-028', 'A-002', 'A-029', 'A-015', 'B-003', 'B-029', 'B-012', 'B-016'],
    ['A-005', 'A-021', 'A-001', 'A-025', 'B-026', 'B-004', 'B-014', 'B-008'],
    ['A-024', 'A-003', 'A-010', 'A-016', 'B-007', 'B-028', 'B-021', 'B-015'],
    ['A-026', 'A-030', 'A-008', 'A-009', 'B-005', 'B-001', 'B-023', 'B-022']
]

# --- Build channel name → index mapping ---
channel_names = [ch['native_channel_name'] for ch in header['amplifier_channels']]
channel_grid = np.zeros((8, 8), dtype=int)

for i in range(8):
    for j in range(8):
        name = layout_names[i][j]
        if name in channel_names:
            channel_grid[i, j] = channel_names.index(name)
        else:
            raise ValueError(f"Channel name '{name}' not found in amplifier_channels.")

# --- Power computation helper ---
def compute_power_window(signal_slice):
    return np.mean(signal_slice**2, axis=1)

# --- Setup animation ---
window_size = 100
step_size = 50
n_samples = amplifier_data.shape[1]
n_frames = (n_samples - window_size) // step_size

fig, ax = plt.subplots()
img = ax.imshow(np.zeros((8, 8)), cmap='viridis', vmin=0, vmax=1000)
cb = plt.colorbar(img, ax=ax)
cb.set_label("Power (µV²)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def make_update_function(lowpass, highpass):
    def update(frame_idx):
        start = frame_idx * step_size
        end = start + window_size
        if end > n_samples:
            return []

        power = compute_power_window(amplifier_data[:, start:end])
        power_grid = power[channel_grid]
        img.set_data(power_grid)
        ax.set_title(f"Time: {start / sample_rate:.2f}s - bandpass:{lowpass}Hz-{highpass}Hz")
        return [img]
    return update

def make_update_save_function(lowpass, highpass):
    def update_save(frame_idx):
        start = frame_idx * step_size
        end = start + window_size
        if end > n_samples:
            return []

        power = compute_power_window(amplifier_data[:, start:end])
        power_grid = power[channel_grid]
        img_save.set_data(power_grid)
        ax_save.set_title(f"Time: {start / sample_rate:.2f}s - bandpass:{lowpass}Hz-{highpass}Hz")
        return [img_save]
    return update_save

# === FULL ANIMATION FOR DISPLAY ===
ani = animation.FuncAnimation(
    fig, make_update_function(lowcut, highcut),
    frames=n_frames, blit=False, interval=50
)

# === LIMITED ANIMATION FOR SAVING ===
# Create a hidden figure (so it doesn't interfere)
fig_save, ax_save = plt.subplots()
img_save = ax_save.imshow(np.zeros((8, 8)), cmap='viridis', vmin=0, vmax=1000)
cb_save = plt.colorbar(img_save, ax=ax_save)
cb_save.set_label("Power (µV²)")
ax_save.set_xlabel("X")
ax_save.set_ylabel("Y")

# 10 seconds × 20 fps = 200 frames
frames_to_save = min(200, n_frames)
ani_save = animation.FuncAnimation(
    fig_save, make_update_save_function(lowcut, highcut),
    frames=frames_to_save, blit=False, interval=50
)

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