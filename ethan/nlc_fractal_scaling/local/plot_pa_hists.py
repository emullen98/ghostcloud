import numpy as np
import matplotlib.pyplot as plt

data = np.load('../../Datasets/NLC_data/find_pl_mc_30G_areas_combined.npy')
exps = data[:, 2]
xmins = data[:, 0]
xmaxs = data[:, 1]

fig, ax = plt.subplots(1, 3)
ax[0].plot(exps, '.')
ax[1].plot(xmins, '.')
ax[2].plot(xmaxs, '.')
plt.show()
