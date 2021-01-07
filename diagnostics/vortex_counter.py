import os.path
import pickle as pickle
import sys

import h5py
import numpy as np

import include.vortex
import include.vortex_detection as vd

"""Calculates the VortexMap for each frame of data and saves the result to a new pickled dataset."""

# Import dataset:
filename = input('Enter filename of the main dataset: ')
v_filepath = '../data/vortexData/{}_VD.pkl'.format(filename)    # Filename for vortex data
data = h5py.File('../data/{}.hdf5'.format(filename), 'r')

# Load in data:
x, y = data['grid/x'][...], data['grid/y'][...]
X, Y = np.meshgrid(x, y)
Nx, Ny = len(x), len(y)
dx, dy = x[1] - x[0], y[1] - y[0]

psi_1 = data['wavefunction/psi_1']
psi_2 = data['wavefunction/psi_2']

# Calculate necessary variables:
n0 = dx * dy * np.sum(abs(psi_1[:, :, 0]) ** 2 + abs(psi_2[:, : 0]) ** 2)
eps = 0.4   # Threshold percentage

# Check if vortex dataset already exists and exit if it does:
try:
    if os.path.isfile(v_filepath):
        raise OSError
except OSError:
    exit('{}: Vortex data file already exists.'.format(sys.exc_info()[0]))

# Start detection:
for i in range(psi_1.shape[-1]):
    # Find positions where density falls below certain threshold
    dens_1_x, dens_1_y = np.where(abs(psi_1[:, :, i]) ** 2 < eps * n0)
    dens_2_x, dens_2_y = np.where(abs(psi_2[:, :, i]) ** 2 < eps * n0)

    v_map = include.vortex.VortexMap()  # Generate empty vortex map

    vd.detect(v_map, dens_1_x, dens_1_y, psi_1[:, :, i], x, y, '1')
    vd.detect(v_map, dens_2_x, dens_2_y, psi_2[:, :, i], x, y, '2')

    v_map.identify_vortices(threshold=2)

    with open(v_filepath, 'ab') as output:
        pickle.dump(v_map, output)
