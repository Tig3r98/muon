import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
num_bins = 20  # adjust as needed

# Get all .txt files with numeric names
file_names = sorted(
    [f for f in os.listdir() if f.endswith('.txt') and f[:-4].isdigit()],
    key=lambda x: int(x[:-4])
)

# Prepare data for 3D plotting
hist_data = []
file_indices = []
bin_edges = None

for idx, file in enumerate(file_names):
    data = np.loadtxt(file, usecols=0)  # read only first column
    counts, bins = np.histogram(data, bins=num_bins)
    hist_data.append(counts)
    file_indices.append(int(file[:-4]))  # filename as number
    if bin_edges is None:
        bin_edges = 0.5 * (bins[:-1] + bins[1:])  # bin centers

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each histogram
dx = 10
dy = 50
for i, (file_index, counts) in enumerate(zip(file_indices, hist_data)):
    xs = np.full_like(bin_edges, file_index)
    ys = bin_edges
    zs = np.zeros_like(counts)
    ax.bar3d(xs, ys, zs, dx, dy, counts, shade=True, alpha=0.5)

ax.set_xlabel('Tensione (V)')
ax.set_ylabel('ADC')
ax.set_zlabel('Conteggi')
#ax.set_title('3D Histogram of First Column Values per File')

ax.view_init(elev=30, azim=135)

plt.tight_layout()
plt.savefig("3d.png")
#plt.show()

