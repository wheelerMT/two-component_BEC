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
filename_1 = 'HQV_grid_gamma=08'
corr_1 = h5py.File('../data/correlations/{}_corr.hdf5'.format(filename_1), 'r')

g_theta_1 = corr_1['g_theta']
g_phi_1 = corr_1['g_phi']

filename_2 = 'HQV_grid_gamma=06'
corr_2 = h5py.File('../data/correlations/{}_corr.hdf5'.format(filename_2), 'r')

g_theta_2 = corr_2['g_theta']
g_phi_2 = corr_2['g_phi']

filename_3 = 'HQV_grid_gamma=03'
corr_3 = h5py.File('../data/correlations/{}_corr.hdf5'.format(filename_3), 'r')

g_theta_3 = corr_3['g_theta']
g_phi_3 = corr_3['g_phi']

time = np.arange(200, 200200, 200)

# * Need to calculate L_theta and L_phi by finding point where G_theta and G_phi fall to 1/4 of their peak
l_theta_1 = np.empty(g_theta_1.shape[-1])
l_phi_1 = np.empty(g_phi_1.shape[-1])
l_theta_2 = np.empty(g_theta_2.shape[-1])
l_phi_2 = np.empty(g_phi_2.shape[-1])
l_theta_3 = np.empty(g_theta_3.shape[-1])
l_phi_3 = np.empty(g_phi_3.shape[-1])

for i in range(g_theta_1.shape[-1]):
    l_theta_1[i] = np.argmax(g_theta_1[:, i] < g_theta_1[0, i] / 4)
    l_phi_1[i] = np.argmax(g_phi_1[:, i] < g_phi_1[0, i] / 4)
    l_theta_2[i] = np.argmax(g_theta_2[:, i] < g_theta_2[0, i] / 4)
    l_phi_2[i] = np.argmax(g_phi_2[:, i] < g_phi_2[0, i] / 4)
    l_theta_3[i] = np.argmax(g_theta_3[:, i] < g_theta_3[0, i] / 4)
    l_phi_3[i] = np.argmax(g_phi_3[:, i] < g_phi_3[0, i] / 4)

# l_theta_1 = running_mean(l_theta_1, 10)
# l_phi_1 = running_mean(l_phi_1, 10)
# l_theta_2 = running_mean(l_theta_2, 10)
# l_phi_2 = running_mean(l_phi_2, 10)

fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_ylabel(r'$L(t)$')
ax.set_xlabel(r'$t/\tau$')

ax.loglog(time[:], l_phi_1[:], linestyle='-', marker='D', markersize=2, color='m', label=r'$L_\phi : \gamma=0.8$')
ax.loglog(time[:], l_theta_1[:], linestyle='-', marker='D', markersize=2, color='g', label=r'$L_\theta : \gamma=0.8$')
ax.loglog(time[:], l_phi_2[:], linestyle='-', marker='D', markersize=2, color='r', label=r'$L_\phi : \gamma=0.6$')
ax.loglog(time[:], l_theta_2[:], linestyle='-', marker='D', markersize=2, color='b', label=r'$L_\theta : \gamma=0.6$')
ax.loglog(time[:], l_phi_3[:], linestyle='-', marker='D', markersize=2, color='y', label=r'$L_\phi : \gamma=0.3$')
ax.loglog(time[:], l_theta_3[:], linestyle='-', marker='D', markersize=2, color='c', label=r'$L_\theta : \gamma=0.3$')

ax.loglog(time[30:], 5.2 * time[30:] ** (1/5), 'k--', label=r'$t^{\frac{1}{5}}$')
ax.loglog(time[30:], 8.5 * time[30:] ** (1/5), 'k--')
ax.legend()
plt.tight_layout()
plt.savefig('../../plots/twoComponent/paper/correlation_lengths.eps', bbox_inches='tight')
plt.show()
