import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# ------------------------------------------------------------------------------------------------------------------
# Load in data
# ------------------------------------------------------------------------------------------------------------------
filename = input('Enter data filename: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r')

psi_1 = data_file['wavefunction/psi_1']
psi_2 = data_file['wavefunction/psi_2']

# Other variables:
x, y = data_file['grid/x'], data_file['grid/y']
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:])

# Loading time variables:
num_of_frames = psi_1.shape[-1]

# ------------------------------------------------------------------------------------------------------------------
# Set up initial figure
# ------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 10))
ax[0].set_ylabel(r'$y / \xi$')
ax[0].set_title(r'$|\psi_1|^2$')
ax[1].set_title(r'$|\psi_2|^2$')
for axis in ax:
    axis.set_xlabel(r'$x / \xi$')
    axis.set_aspect('equal')

cvals_dens = np.linspace(0, 3.2e9 / (1024 ** 2), 25)

# Initial frame plot:
densPlus_plot = ax[0].contourf(X, Y, abs(psi_1[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
densMinus_plot = ax[1].contourf(X, Y, abs(psi_2[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
plt.colorbar(densMinus_plot, ax=ax[1], fraction=0.045)
plt.show()
