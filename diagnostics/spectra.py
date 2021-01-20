import h5py
import numpy as np
import numexpr as ne
import pyfftw
from numpy.fft import fftshift, ifftshift
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def spectral_derivative(array, wvn_x, wvn_y, f_fft2, f_ifft2):
    return f_ifft2(ifftshift(1j * wvn_x * fftshift(f_fft2(array)))), \
           f_ifft2(ifftshift(1j * wvn_y * fftshift(f_fft2(array))))


# ------------------------------------------------------------------------------------------------------------------
# Loading required data
# ------------------------------------------------------------------------------------------------------------------
filename = input('Enter name of data file: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r')

# Grid data:
x, y = np.array(data_file['grid/x']), np.array(data_file['grid/y'])
Nx, Ny = x.size, y.size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)  # K-space spacing
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)
K = ne.evaluate("sqrt(Kx ** 2 + Ky ** 2)")
wvn = kxx[Nx // 2:]

# Wavefunction data:
frame = int(input('Enter frame number to construct spectra for: '))
psi_1 = data_file['wavefunction/psi_1'][:, :, frame]
psi_2 = data_file['wavefunction/psi_2'][:, :, frame]
n = abs(psi_1) ** 2 + abs(psi_2) ** 2

# Build FFT plan:
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex64')
fft2 = pyfftw.builders.fft2(wfn_data)
ifft2 = pyfftw.builders.ifft2(wfn_data)

# ------------------------------------------------------------------------------------------------------------------
# Calculate generalised velocities and occupation number
# ------------------------------------------------------------------------------------------------------------------
# Total occupation number n(k):
occupation = fftshift(fft2(psi_1) * np.conj(fft2(psi_1)) + fft2(psi_2) * np.conj(fft2(psi_2))).real / (Nx * Ny)

# ------------------------------------
# Quantum pressure, w_q
# ------------------------------------
grad_sqrtn_x, grad_sqrtn_y = spectral_derivative(np.sqrt(n), Kx, Ky, fft2, ifft2)
wq_x = fftshift(fft2(grad_sqrtn_x)) / np.sqrt(Nx * Ny)
wq_y = fftshift(fft2(grad_sqrtn_y)) / np.sqrt(Nx * Ny)

# ------------------------------------
# Weighted velocity, w_i & w_c:
# ------------------------------------
# Calculate mass velocity:
dpsi_1_x, dpsi_1_y = spectral_derivative(psi_1, Kx, Ky, fft2, ifft2)
dpsi_2_x, dpsi_2_y = spectral_derivative(psi_2, Kx, Ky, fft2, ifft2)
v_x = (np.conj(psi_1) * dpsi_1_x - np.conj(dpsi_1_x) * psi_1 + np.conj(psi_2) * dpsi_2_x - np.conj(dpsi_2_x) * psi_2) \
       / (2 * 1j * n)
v_y = (np.conj(psi_1) * dpsi_1_y - np.conj(dpsi_1_y) * psi_1 + np.conj(psi_2) * dpsi_2_y - np.conj(dpsi_2_y) * psi_2) \
       / (2 * 1j * n)

# FFT of weighted velocity fields:
u_x = fftshift(fft2(np.sqrt(n) * v_x)) / np.sqrt(Nx * Ny)
u_y = fftshift(fft2(np.sqrt(n) * v_y)) / np.sqrt(Nx * Ny)

# Coefficients of incompressible and compressible velocities:
A_1 = 1 - Kx ** 2 / K ** 2
A_2 = 1 - Ky ** 2 / K ** 2
B = Kx * Ky / K ** 2
C_1 = Kx ** 2 / K ** 2
C_2 = Ky ** 2 / K ** 2

# Incompressible:
ui_x = A_1 * u_x - B * u_y
ui_y = -B * u_x + A_2 * u_y

# Compressible:
uc_x = C_1 * u_x + B * u_y
uc_y = B * u_x + C_2 * u_y

# ------------------------------------------------------------------------------------------------------------------
# Calculate energies
# ------------------------------------------------------------------------------------------------------------------
# Quantum pressure:
E_q = 0.5 * (abs(wq_x) ** 2 + abs(wq_y) ** 2)

# Incompressible:
E_vi = 0.5 * (abs(ui_x) ** 2 + abs(ui_y) ** 2)

# Compressible:
E_vc = 0.5 * (abs(uc_x) ** 2 + abs(uc_y) ** 2)

# ------------------------------------------------------------------------------------------------------------------
# Calculating energy spectrum
# ------------------------------------------------------------------------------------------------------------------
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)

centerx = Nx // 2
centery = Ny // 2

eps = 1e-50  # Voids log(0)

# Defining zero arrays for spectra:
e_occ = np.zeros((box_radius, )) + eps
e_q = np.zeros((box_radius, )) + eps
e_vi = np.zeros((box_radius, )) + eps
e_vc = np.zeros((box_radius, )) + eps

nc = np.zeros((box_radius, ))  # Counts the number of times we sum over a given shell

for kx in range(Nx):
    for ky in range(Ny):
        k = int(np.ceil(np.sqrt((kx - centerx) ** 2 + (ky - centery) ** 2)))
        nc[k] += 1

        e_occ[k] += occupation[kx, ky]
        e_q[k] += 2 * K[kx, ky] ** (-2) * E_q[kx, ky]
        e_vi[k] += 2 * K[kx, ky] ** (-2) * E_vi[kx, ky]
        e_vc[k] += 2 * K[kx, ky] ** (-2) * E_vc[kx, ky]

e_occ[:] /= (nc[:] * dkx)
e_q[:] /= (nc[:] * dkx)
e_vi[:] /= (nc[:] * dkx)
e_vc[:] /= (nc[:] * dkx)

plt.loglog(wvn, e_occ[:Nx//2], color='k', marker='D')
plt.loglog(wvn, e_q[:Nx//2], color='m', marker='D')
plt.loglog(wvn, e_vi[:Nx//2], color='r', marker='D')
plt.loglog(wvn, e_vc[:Nx//2], color='b', marker='D')
plt.show()
