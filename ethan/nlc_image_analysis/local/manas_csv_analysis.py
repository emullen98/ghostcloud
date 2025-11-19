import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import sys
import os
sys.path.append('/Users/emullen98/Desktop')
import helper_scripts as hs

pwd = hs.get_pwd()
path = '/Users/emullen98/Downloads/pngs/png_argmax' 
files_by_hour = [file for file in os.listdir(path) if file.endswith('.csv') and 'multi' not in file]

# Extract times and sort files by time
files_with_times = []
for file in files_by_hour:
    # Extract time string (e.g., "1100-1200")
    parts = file.split('_')
    time_str = None
    for part in parts:
        if '-' in part and len(part) == 9:  # Format: "HHMM-HHMM"
            time_str = part
            break
    
    if time_str:
        # Convert start time to minutes for sorting/coloring
        start_time = time_str.split('-')[0]
        hours = int(start_time[:2])
        minutes = int(start_time[2:])
        time_minutes = hours * 60 + minutes
        files_with_times.append((file, time_str, time_minutes))

# Sort by time
files_with_times.sort(key=lambda x: x[2])

# Create colormap
cmap = plt.cm.viridis  # You can use 'plasma', 'inferno', 'cividis', etc.
norm = Normalize(vmin=min(f[2] for f in files_with_times), vmax=max(f[2] for f in files_with_times))

fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

for file, time_str, time_minutes in files_with_times:
    color = cmap(norm(time_minutes))

    df = pd.read_csv(os.path.join(path, file))
    areas = df['area_px'].to_numpy()
    perims = df['perim_px'].to_numpy()
    
    areas_cx, areas_cy = hs.ccdf(areas)
    perims_cx, perims_cy = hs.ccdf(perims)

    ax[0].plot(areas_cx, areas_cy, label=time_str, color=color)
    ax[1].plot(perims_cx, perims_cy, label=time_str, color=color)

x_area, y_area = hs.linemaker(-1, [10**5, 3*10**(-2)], 10**3, 10**6)
x_perim, y_perim = hs.linemaker(-1.5, [2000, 0.2], 300, 30000)

ax[0].plot(x_area, y_area, 'k--', label='$C(A) \sim A^{-1}$')
ax[1].plot(x_perim, y_perim, 'k--', label='$C(P) \sim P^{-1.5}$')

ax[0].set_title('Area CCDF')
ax[0].loglog()
ax[0].legend()
ax[1].set_title('Perimeter CCDF')
ax[1].loglog()
ax[1].legend()
fig.suptitle('PNG argmax all time windows')

# Add colorbar to show time mapping
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[1])
cbar.set_label('Time (minutes from midnight)')

fig.savefig(f'{pwd}/temp.png')  