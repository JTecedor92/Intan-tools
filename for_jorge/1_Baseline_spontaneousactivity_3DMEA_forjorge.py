import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from intanutil.data import (
    calculate_data_size,
    read_all_data_blocks,
    check_end_of_file,
    parse_data,
    data_to_result
)
from intanutil.filter import apply_notch_filter

# --- Step 1: Open File Dialog ---
root = tk.Tk()
root.withdraw()  # Hide root window

file_path = filedialog.askopenfilename(
    title="Select a RHS File",
    filetypes=[("RHS files", "*.rhs"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected.")
    exit()

print("Selected file path:", file_path)
file_name = os.path.basename(file_path)
print("File name:", file_name)
print("Directory:", os.getcwd())

# --- Step 2: Read Header Using read_header() ---
from intanutil.header import read_header  # Replace with correct path if needed

with open(file_path, 'rb') as fid:
    header = read_header(fid)

    data_present, filesize, num_blocks, num_samples = calculate_data_size(header, file_path, fid)
    data = read_all_data_blocks(header, num_samples, num_blocks, fid)
    check_end_of_file(filesize, fid)

# --- Step 3: Process & Parse Data ---
apply_notch_filter(header, data)
parse_data(header, data)
result = data_to_result(header, data, {})

amplifier_data = result['amplifier_data'][:64]
sample_rate = header['sample_rate']

# --- Step 4: Layout Mapping ---
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

channel_names = [ch['native_channel_name'] for ch in header['amplifier_channels']]
channel_grid = np.zeros((8, 8), dtype=int)

for i in range(8):
    for j in range(8):
        name = layout_names[i][j]
        if name in channel_names:
            channel_grid[i, j] = channel_names.index(name)
        else:
            raise ValueError(f"Channel name '{name}' not found in amplifier_channels.")

# --- Step 5: Animation ---
def compute_power_window(signal_slice):
    return np.mean(signal_slice**2, axis=1)

window_size = 100
step_size = 50
n_samples = amplifier_data.shape[1]
n_frames = (n_samples - window_size) // step_size

fig, ax = plt.subplots()
img = ax.imshow(np.zeros((8, 8)), cmap='viridis', vmin=0, vmax=1000)
cb = plt.colorbar(img, ax=ax)
cb.set_label("Power (µV²)")
ax.set_title("Channel Power Over Time")
ax.set_xlabel("X")
ax.set_ylabel("Y")



def update(frame_idx):
    start = frame_idx * step_size
    end = start + window_size
    if end > n_samples:
        return []

    power = compute_power_window(amplifier_data[:, start:end])
    power_grid = power[channel_grid]
    img.set_data(power_grid)
    ax.set_title(f"Time: {start / sample_rate:.2f}s")
    return [img]

ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=50)
plt.show()
