import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# Calculate correlation lengths
l_theta = np.empty(g_theta.shape[-1])
l_phi = np.empty(g_phi.shape[-1])
for i in range(g_theta.shape[-1]):
    l_theta[i] = np.argmax(g_theta[:, i] < g_theta[0, i] / 4)
    l_phi[i] = np.argmax(g_phi[:, i] < g_phi[0, i] / 4)

time = np.arange(200, 200000, 200)

# Plots:
colors = ['r', 'g', 'b', 'y', 'c', 'm']
markers = ['D', 'o', '*', 'v', 'X', 's']
frames = [25, 75, 150, 250, 525, 850]
radius = 130 * 2
step = 4

fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8 * 2), sharex=True)
axins_spin = inset_axes(ax[0], width="40%", height="40%")
axins_spin.set_xlim(0, 4)
axins_spin.set_ylabel(r'$G_\phi(r/L_\phi, t)$')
axins_spin.set_xlabel(r'$r/L_\phi$')
axins_mass = inset_axes(ax[1], width="40%", height="40%")
axins_mass.set_xlim(0, 4)
axins_mass.set_ylabel(r'$G_\theta(r/L_\theta, t)$')
axins_mass.set_xlabel(r'$r/L_\theta$')
ax[0].set_ylabel(r'$G_\phi(r, t)$')
ax[1].set_ylabel(r'$G_\theta(r, t)$')
ax[1].set_xlabel(r'$r/\xi_d$')

for i in range(6):
    ax[0].plot(np.arange(0, radius / 2, step), g_phi[:radius // 2:step, frames[i] - 1], linestyle='-',
               marker=markers[i], markersize=4, color=colors[i],
               label=r'$t = {:.0f} \times 10^3\tau$'.format(frames[i] * 200 / 1000))
    ax[1].plot(np.arange(0, radius / 2, step), g_theta[:radius // 2:step, frames[i] - 1], linestyle='-',
               marker=markers[i], markersize=4, color=colors[i])
    axins_spin.plot(np.arange(0, g_phi.shape[-1] / 2, step) / l_phi[frames[i] - 1], g_phi[:g_phi.shape[-1] // 2:step, frames[i] - 1],
                    linestyle='--', marker=markers[i], markersize=2, color=colors[i])
    axins_mass.plot(np.arange(0, g_phi.shape[-1] / 2, step) / l_theta[frames[i] - 1], g_theta[:g_phi.shape[-1] // 2:step, frames[i] - 1],
                    linestyle='--', marker=markers[i], markersize=2, color=colors[i])

ax[0].legend(loc=3, fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig('../../plots/twoComponent/paper/correlations.eps', bbox_inches='tight')
plt.show()
