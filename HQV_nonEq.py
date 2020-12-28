import numpy as np
import cupy as cp
import h5py
import include.symplecticMethod as sm

"""Generates HQVs in both components of a two-component BEC from a non-equilibrium initial condition."""

# --------------------------------------------------------------------------------------------------------------------
# Controlled variables:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 1024, 1024
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
g1 = 3e-5
g2 = 3e-5
gamma = 0.75
g12 = g1 * gamma

# Time steps, number and wavefunction save variables
Nt = 10000000
Nframe = 10000  # Saves data every Nframe time steps
dt = 1e-2  # Timestep
t = 0.
save_index = 0   # Array index

# --------------------------------------------------------------------------------------------------------------------
# Generate the initial state:
# --------------------------------------------------------------------------------------------------------------------
atom_num = 1.6e9 / (Nx * Ny)

# Draw values of theta from Gaussian distribution:
np.random.seed(9973)
theta_k_1 = cp.asarray(np.random.uniform(low=0, high=1, size=(Nx, Ny)) * 2 * np.pi)
np.random.seed(9962)
theta_k_2 = cp.asarray(np.random.uniform(low=0, high=1, size=(Nx, Ny)) * 2 * np.pi)

# Generate density array so that only certain modes are occupied:
n_k = cp.zeros((Nx, Ny))
n_k[Nx // 2, Ny // 2] = atom_num * 0.5 / (dx * dy)  # k = (0, 0)
n_k[Nx // 2 + 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (1, 0)
n_k[Nx // 2 + 1, Ny // 2 + 1] = atom_num * 0.125 / (dx * dy)  # k = (1, 1)
n_k[Nx // 2 - 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (-1, 0)
n_k[Nx // 2 - 1, Ny // 2 - 1] = atom_num * 0.125 / (dx * dy)  # k = (-1, -1)

psi_1_k = cp.fft.fftshift(Nx * Ny * cp.sqrt(n_k) * cp.exp(1j * theta_k_1)) / cp.sqrt(2.)
psi_2_k = cp.fft.fftshift(Nx * Ny * cp.sqrt(n_k) * cp.exp(1j * theta_k_2)) / cp.sqrt(2.)

# ------------------------------------------------------------------------------------------------------------------
# Creating save file and saving initial data
# ------------------------------------------------------------------------------------------------------------------
filename = 'HQV_nonEq'    # Name of file to save data to
data_path = 'data/{}.hdf5'.format(filename)

with h5py.File(data_path, 'w') as data:
    # Saving spatial data:
    data.create_dataset('grid/x', x.shape, data=cp.asnumpy(x))
    data.create_dataset('grid/y', y.shape, data=cp.asnumpy(y))

    # Saving time variables:
    data.create_dataset('time/Nt', data=Nt)
    data.create_dataset('time/dt', data=dt)
    data.create_dataset('time/Nframe', data=Nframe)

    # Creating empty wavefunction datasets to store data:
    data.create_dataset('wavefunction/psi_1', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
    data.create_dataset('wavefunction/psi_2', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')

    # Stores initial state:
    data.create_dataset('initial_state/psi_1', data=cp.asnumpy(cp.fft.ifft2(psi_1_k)))
    data.create_dataset('initial_state/psi_2', data=cp.asnumpy(cp.fft.ifft2(psi_2_k)))

# ------------------------------------------------------------------------------------------------------------------
# Real time evolution
# ------------------------------------------------------------------------------------------------------------------
for i in range(Nt):
    # Kinetic step:
    sm.kinetic_evolution(psi_1_k, dt, Kx, Ky)
    sm.kinetic_evolution(psi_2_k, dt, Kx, Ky)

    psi_1 = cp.fft.ifft2(psi_1_k)
    psi_2 = cp.fft.ifft2(psi_2_k)

    # Potential step:
    psi_1, psi_2 = sm.potential_evolution(psi_1, psi_2, dt, g1, g2, g12)

    psi_1_k = cp.fft.fft2(psi_1)
    psi_2_k = cp.fft.fft2(psi_2)

    # Kinetic step:
    sm.kinetic_evolution(psi_1_k, dt, Kx, Ky)
    sm.kinetic_evolution(psi_2_k, dt, Kx, Ky)

    # Saves data
    if np.mod(i + 1, Nframe) == 0:
        with h5py.File(data_path, 'r+') as data:
            new_psi_1 = data['wavefunction/psi_1']
            new_psi_2 = data['wavefunction/psi_2']
            new_psi_1.resize((Nx, Ny, save_index + 1))
            new_psi_2.resize((Nx, Ny, save_index + 1))
            new_psi_1[:, :, save_index] = cp.asnumpy(cp.fft.ifft2(psi_1_k))
            new_psi_2[:, :, save_index] = cp.asnumpy(cp.fft.ifft2(psi_2_k))
        save_index += 1

    # Prints current time
    if np.mod(i, Nframe) == 0:
        print('t = %1.4f' % t)

    t += dt
