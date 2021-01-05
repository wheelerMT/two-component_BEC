import h5py
import numpy as np
import include.vortex
import include.vortex_detection as vd
import pickle as pickle

"""Calculates the number of vortices in each frame of data and saves the result to a new dataset."""

# Import dataset:
filename = 'HQV_imp_Nvort=200'
data = h5py.File('../data/{}.hdf5'.format(filename), 'r')

# Load in data:
x, y = data['grid/x'][...], data['grid/y'][...]
X, Y = np.meshgrid(x, y)
Nx, Ny = len(x), len(y)
dx, dy = x[1] - x[0], y[1] - y[0]

psi_1 = data['wavefunction/psi_1']
psi_2 = data['wavefunction/psi_2']

# Calculate necessary variables:
n0 = 1.6e9 / (Nx * Ny)  # Background density
eps = 0.1   # Threshold percentage

maps = []

# Start detection:
for i in range(psi_1.shape[-1]):
    # Find positions where density falls below certain threshold
    dens_1_x, dens_1_y = np.where(abs(psi_1[:, :, i]) ** 2 < eps * n0)
    dens_2_x, dens_2_y = np.where(abs(psi_2[:, :, i]) ** 2 < eps * n0)

    v_map = include.vortex.VortexMap()  # Generate empty vortex map

    vd.detect(v_map, dens_1_x, dens_1_y, psi_1[:, :, i], x, y, '1')
    vd.detect(v_map, dens_2_x, dens_2_y, psi_2[:, :, i], x, y, '2')

    v_map.identify_vortices(threshold=2)

    with open('../data/vortexData/{}_VD.pkl'.format(filename), 'wb+') as output:
        pickle.dump(v_map, output, pickle.HIGHEST_PROTOCOL)


r"""
used_map = maps[2]

# Plot the vortex overlay for this map
fig, ax = plt.subplots(2, figsize=(8, 8))
for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.set_ylim(y.min(), y.max())

ax[0].contourf(X, Y, abs(psi_1[:, :, 2]) ** 2, levels=25, cmap='gnuplot')
ax[1].contourf(X, Y, abs(psi_2[:, :, 2]) ** 2, levels=25, cmap='gnuplot')

for vortex in used_map.vortices_hqv:
    if vortex.winding > 0:
        if vortex.component == '1':
            ax[0].plot(*zip(vortex.get_coords()), 'w^')
        if vortex.component == '2':
            ax[1].plot(*zip(vortex.get_coords()), 'ws')
    if vortex.winding < 0:
        if vortex.component == '1':
            ax[0].plot(*zip(vortex.get_coords()), 'k^')
        if vortex.component == '2':
            ax[1].plot(*zip(vortex.get_coords()), 'ks')

plt.show()
"""
