import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 13})


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# Import correlation data:
data = 'HQV_grid_gamma=06'
corr = h5py.File('../data/correlations/{}_corr.hdf5'.format(data), 'r')

g_theta = corr['g_theta']
g_phi = corr['g_phi']

time = np.arange(200, 200000, 200)

# Plots:
colors = ['r', 'g', 'b', 'y', 'c', 'm']
markers = ['D', 'o', '*', 'v', 'X', 's']
frames = [25, 75, 150, 250, 500, 900]
radius = 130 * 2
step = 6

fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8 * 2), sharex=True)

ax[0].set_ylabel(r'$G_\phi(r, t)$')
ax[1].set_ylabel(r'$G_\theta(r, t)$')
ax[1].set_xlabel(r'$r/\xi_d$')

for i in range(6):
    ax[0].plot(np.arange(0, radius / 2, step), g_phi[:radius//2:step, frames[i] - 1], linestyle='-', marker=markers[i],
               markersize=4, color=colors[i], label=r'$t = {} \times 10^3t_s$'.format(frames[i] * 200 / 1000))
    ax[1].plot(np.arange(0, radius / 2, step), g_theta[:radius//2:step, frames[i] - 1], linestyle='-', marker=markers[i],
               markersize=4, color=colors[i])

ax[0].legend()
plt.tight_layout()
plt.savefig('../../plots/twoComponent/paper/correlations.eps', bbox_inches='tight')
plt.show()
