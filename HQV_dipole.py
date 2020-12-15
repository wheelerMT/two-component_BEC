import numpy as np
import cupy as cp
import h5py
from include.phaseImprinting import get_phase
import include.symplecticMethod as sm
import matplotlib.pyplot as plt

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
n0 = cp.ones(shape=(Nx, Ny))  # Background density

# Generate phase:
N_vort = 2  # One dipole
positions = [-32, -32, 32, -32]
theta = get_phase(N_vort, positions, Nx, Ny, cp.asnumpy(X), cp.asnumpy(Y), len_x, len_y)

# Generate initial wavefunctions:
psi_1 = cp.sqrt(n0 / 2) * cp.exp(1j * cp.asarray(theta))
psi_2 = cp.sqrt(n0 / 2)
psi_1_k = cp.fft.fft2(psi_1)
psi_2_k = cp.fft.fft2(psi_2)

# Getting atom number for renormalisation in imaginary time evolution
atom_num_1 = dx * dy * cp.sum(cp.abs(psi_1) ** 2)
atom_num_2 = dx * dy * cp.sum(cp.abs(psi_2) ** 2)

# Phase of initial state to allow fixing of phase during imaginary time evolution
theta_fix_1 = np.angle(psi_1)
theta_fix_2 = np.angle(psi_2)

# ------------------------------------------------------------------------------------------------------------------
# Imaginary time evolution
# ------------------------------------------------------------------------------------------------------------------
for i in range(2000):
    # Kinetic step:
    sm.kinetic_evolution(psi_1_k, -1j * dt, Kx, Ky)
    sm.kinetic_evolution(psi_2_k, -1j * dt, Kx, Ky)

    psi_1 = cp.fft.ifft2(psi_1_k)
    psi_2 = cp.fft.ifft2(psi_2_k)

    # Potential step:
    psi_1, psi_2 = sm.potential_evolution(psi_1, psi_2, -1j * dt, g1, g2, g12)

    psi_1_k = cp.fft.fft2(psi_1)
    psi_2_k = cp.fft.fft2(psi_2)

    # Kinetic step:
    sm.kinetic_evolution(psi_1_k, -1j * dt, Kx, Ky)
    sm.kinetic_evolution(psi_2_k, -1j * dt, Kx, Ky)

    # Re-normalising:
    atom_num_new_1 = dx * dy * cp.sum(cp.abs(cp.fft.ifft2(psi_1_k)) ** 2)
    atom_num_new_2 = dx * dy * cp.sum(cp.abs(cp.fft.ifft2(psi_2_k)) ** 2)
    psi_1_k = cp.fft.fft2(cp.sqrt(atom_num_1) * cp.fft.ifft2(psi_1_k) / cp.sqrt(atom_num_new_1))
    psi_2_k = cp.fft.fft2(cp.sqrt(atom_num_2) * cp.fft.ifft2(psi_2_k) / cp.sqrt(atom_num_new_2))

    # Fixing phase:
    psi_1 = cp.fft.ifft2(psi_1_k)
    psi_2 = cp.fft.ifft2(psi_2_k)
    psi_1 *= cp.exp(1j * theta_fix_1) / cp.exp(1j * cp.angle(psi_1))
    psi_2 *= cp.exp(1j * theta_fix_2) / cp.exp(1j * cp.angle(psi_2))
    psi_1_k = cp.fft.fft2(psi_1)
    psi_2_k = cp.fft.fft2(psi_2)
