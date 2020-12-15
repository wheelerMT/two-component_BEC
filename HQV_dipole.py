import numpy as np
import cupy as cp
import h5py
from include.phaseImprinting import get_phase

"""Creates a HQV dipole in one of the components of the system and evolves the dynamics in real time."""

# --------------------------------------------------------------------------------------------------------------------
# Controlled variables:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 128, 128
Mx, My = Nx // 2, Ny // 2  # Number of grid pts
dx = dy = 1  # Grid spacing
dkx = np.pi / (Mx * dx)
dky = np.pi / (My * dy)  # K-space spacing
len_x = Nx * dx  # Box length
len_y = Ny * dy
x = cp.arange(-Mx, Mx) * dx
y = cp.arange(-My, My) * dy
X, Y = cp.meshgrid(x, y)  # Spatial meshgrid

# k-space arrays and meshgrid:
kx = cp.arange(-Mx, Mx) * dkx
ky = cp.arange(-My, My) * dky
Kx, Ky = cp.meshgrid(kx, ky)  # K-space meshgrid
Kx, Ky = cp.fft.fftshift(Kx), cp.fft.fftshift(Ky)

# Potential and interaction parameters
V = 0.  # Doubly periodic box
g1 = 1
g2 = 1
g12 = 0.5

# Time steps, number and wavefunction save variables
Nt = 100000
Nframe = 1000   # Save data every Nframe number of timesteps
dt = 1e-2  # Imaginary time timestep
t = 0.
save_index = 0   # Array index

filename = 'HQV_dipole'    # Name of file to save data to
data_path = '../scratch/data/scalar/{}.hdf5'.format(filename)
backup_data_path = '../scratch/data/scalar/{}_backup.hdf5'.format(filename)

# --------------------------------------------------------------------------------------------------------------------
# Generate the initial state:
# --------------------------------------------------------------------------------------------------------------------
# Initial state parameters:
n0 = 1.  # Background density

# Generate phase:
N_vort = 2  # One dipole
positions = [-32, -32, 32, -32]
theta = get_phase(N_vort, positions, Nx, Ny, cp.asnumpy(X), cp.asnumpy(Y), len_x, len_y)

# Generate initial wavefunctions:
psi_1 = cp.sqrt(n0 / 2) * cp.exp(1j * cp.asarray(theta))
psi_2 = cp.sqrt(n0 / 2)
