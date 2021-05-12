import h5py
import numpy as np
from tabulate import tabulate
from numpy.fft import fftshift, ifftshift
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 13})


def spectral_derivative(array, wvn_x, wvn_y):
    return np.fft.ifft2(ifftshift(1j * wvn_x * fftshift(np.fft.fft2(array)))), \
           np.fft.ifft2(ifftshift(1j * wvn_y * fftshift(np.fft.fft2(array))))


def calc_gen_nematic_vel(dens, nematic_x, nematic_y):
    return np.sqrt(2 * dens) / 2 * nematic_x, np.sqrt(2 * dens) / 2 * nematic_y


# ------------------------------------------------------------------------------------------------------------------
# Loading required data
# ------------------------------------------------------------------------------------------------------------------
filename = 'frames/200kf_HQV_grid_gamma=06'  # input('Enter name of data file: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r')

# Grid data:
x, y = np.array(data_file['grid/x']), np.array(data_file['grid/y'])
X, Y = np.meshgrid(x, y)
Nx, Ny = x.size, y.size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)  # K-space spacing
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)
K = np.sqrt(Kx ** 2 + Ky ** 2)
wvn = kxx[Nx // 2:]

# Wavefunction data:
# Get info about the saved times of data and prints to screen:
saved_times = data_file['saved_times']
list_of_times = []
for i in range(saved_times.shape[0]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))

frame = -1  # int(input('Enter the frame number you wish to plot: '))

psi_1 = data_file['wavefunction/psi_1'][:, :, frame]
psi_2 = data_file['wavefunction/psi_2'][:, :, frame]

n_1 = abs(psi_1) ** 2
n_2 = abs(psi_2) ** 2

# Build FFT plan:
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2

# ------------------------------------------------------------------------------------------------------------------
# Calculate generalised velocities and occupation number
# ------------------------------------------------------------------------------------------------------------------
# Total occupation number n(k):
occ_1 = fftshift(fft2(psi_1) * np.conj(fft2(psi_1))).real / (Nx * Ny)
occ_2 = fftshift(fft2(psi_2) * np.conj(fft2(psi_2))).real / (Nx * Ny)

# ------------------------------------
# Quantum pressure, w_q
# ------------------------------------
grad_sqrtn_x_1, grad_sqrtn_y_1 = spectral_derivative(np.sqrt(n_1), Kx, Ky)
wq_x_1 = fftshift(fft2(grad_sqrtn_x_1)) / np.sqrt(Nx * Ny)
wq_y_1 = fftshift(fft2(grad_sqrtn_y_1)) / np.sqrt(Nx * Ny)

grad_sqrtn_x_2, grad_sqrtn_y_2 = spectral_derivative(np.sqrt(n_2), Kx, Ky)
wq_x_2 = fftshift(fft2(grad_sqrtn_x_2)) / np.sqrt(Nx * Ny)
wq_y_2 = fftshift(fft2(grad_sqrtn_y_2)) / np.sqrt(Nx * Ny)

# ------------------------------------
# Weighted velocity, w_i & w_c:
# ------------------------------------
# Calculate mass velocity:
dpsi_1_x, dpsi_1_y = spectral_derivative(psi_1, Kx, Ky)
v_x_1 = (np.conj(psi_1) * dpsi_1_x - np.conj(dpsi_1_x) * psi_1) / (2 * 1j * n_1)
v_y_1 = (np.conj(psi_1) * dpsi_1_y - np.conj(dpsi_1_y) * psi_1) / (2 * 1j * n_1)

dpsi_2_x, dpsi_2_y = spectral_derivative(psi_2, Kx, Ky)
v_x_2 = (np.conj(psi_2) * dpsi_2_x - np.conj(dpsi_2_x) * psi_2) / (2 * 1j * n_2)
v_y_2 = (np.conj(psi_2) * dpsi_2_y - np.conj(dpsi_2_y) * psi_2) / (2 * 1j * n_2)

# FFT of weighted velocity fields:
u_x_1 = fftshift(fft2(np.sqrt(n_1) * v_x_1)) / np.sqrt(Nx * Ny)
u_y_1 = fftshift(fft2(np.sqrt(n_1) * v_y_1)) / np.sqrt(Nx * Ny)

u_x_2 = fftshift(fft2(np.sqrt(n_2) * v_x_2)) / np.sqrt(Nx * Ny)
u_y_2 = fftshift(fft2(np.sqrt(n_2) * v_y_2)) / np.sqrt(Nx * Ny)

# Coefficients of incompressible and compressible velocities:
A_1 = 1 - Kx ** 2 / K ** 2
A_2 = 1 - Ky ** 2 / K ** 2
B = Kx * Ky / K ** 2
C_1 = Kx ** 2 / K ** 2
C_2 = Ky ** 2 / K ** 2

# Compressible:
uc_x_1 = C_1 * u_x_1 + B * u_y_1
uc_y_1 = B * u_x_1 + C_2 * u_y_1

uc_x_2 = C_1 * u_x_2 + B * u_y_2
uc_y_2 = B * u_x_2 + C_2 * u_y_2

# Incompressible:
ui_x_1 = u_x_1 - uc_x_1
ui_y_1 = u_y_1 - uc_y_1

ui_x_2 = u_x_2 - uc_x_2
ui_y_2 = u_y_2 - uc_y_2

# ------------------------------------------------------------------------------------------------------------------
# Calculate energies
# ------------------------------------------------------------------------------------------------------------------
# Quantum pressure:
E_q_1 = 0.5 * (abs(wq_x_1) ** 2 + abs(wq_y_1) ** 2)
E_q_2 = 0.5 * (abs(wq_x_2) ** 2 + abs(wq_y_2) ** 2)

# Incompressible:
E_vi_1 = 0.5 * (abs(ui_x_1) ** 2 + abs(ui_y_1) ** 2)
E_vi_2 = 0.5 * (abs(ui_x_2) ** 2 + abs(ui_y_2) ** 2)

# Compressible:
E_vc_1 = 0.5 * (abs(uc_x_1) ** 2 + abs(uc_y_1) ** 2)
E_vc_2 = 0.5 * (abs(uc_x_2) ** 2 + abs(uc_y_2) ** 2)

# ------------------------------------------------------------------------------------------------------------------
# Calculating energy spectrum
# ------------------------------------------------------------------------------------------------------------------
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)

centerx = Nx // 2
centery = Ny // 2

eps = 1e-50  # Voids log(0)

# Defining zero arrays for spectra:
e_occ_1 = np.zeros((box_radius,)) + eps
e_occ_2 = np.zeros((box_radius,)) + eps
e_q_1 = np.zeros((box_radius,)) + eps
e_q_2 = np.zeros((box_radius,)) + eps
e_vi_1 = np.zeros((box_radius,)) + eps
e_vi_2 = np.zeros((box_radius,)) + eps
e_vc_1 = np.zeros((box_radius,)) + eps
e_vc_2 = np.zeros((box_radius,)) + eps

nc = np.zeros((box_radius,))  # Counts the number of times we sum over a given shell

for kx in range(Nx):
    for ky in range(Ny):
        k = int(np.round(np.sqrt((kx - centerx) ** 2 + (ky - centery) ** 2)))
        nc[k] += 1

        e_occ_1[k] += occ_1[kx, ky]
        e_occ_2[k] += occ_2[kx, ky]

        e_q_1[k] += 2 * K[kx, ky] ** (-2) * E_q_1[kx, ky]
        e_q_2[k] += 2 * K[kx, ky] ** (-2) * E_q_2[kx, ky]

        e_vi_1[k] += 2 * K[kx, ky] ** (-2) * E_vi_1[kx, ky]
        e_vi_2[k] += 2 * K[kx, ky] ** (-2) * E_vi_2[kx, ky]

        e_vc_1[k] += 2 * K[kx, ky] ** (-2) * E_vc_1[kx, ky]
        e_vc_2[k] += 2 * K[kx, ky] ** (-2) * E_vc_2[kx, ky]

e_occ_1[:] /= (nc[:] * dkx)
e_occ_2[:] /= (nc[:] * dkx)
e_q_1[:] /= (nc[:] * dkx)
e_q_2[:] /= (nc[:] * dkx)
e_vi_1[:] /= (nc[:] * dkx)
e_vi_2[:] /= (nc[:] * dkx)
e_vc_1[:] /= (nc[:] * dkx)
e_vc_2[:] /= (nc[:] * dkx)

sum_1 = e_q_1 + e_vi_1 + e_vc_1 + e_q_2 + e_vi_2 + e_vc_2

print('{:.10e}'.format(np.sum(2 * np.pi * e_occ_2[:Nx // 2] * wvn[:])))

fig, ax = plt.subplots(1, )

ax.set_ylim(1e2, 1e10)
ax.set_xlabel(r'$ka_s$')
ax.set_ylabel(r'$n(k)$')
# ax.set_ylim(bottom=1e3, top=4e9)

ax.loglog(wvn, e_occ_1[:Nx // 2] + e_occ_2[:Nx // 2], color='k', marker='D', markersize=2, linestyle='None',
          label=r'$n(k)$')
ax.loglog(wvn, e_q_1[:Nx // 2] + e_q_2[:Nx // 2], color='m', marker='D', markersize=2, linestyle='None',
          label=r'$n_q(k)$')
ax.loglog(wvn, e_vi_1[:Nx // 2] + e_vi_2[:Nx // 2], color='r', marker='D', markersize=2, linestyle='None',
          label=r'$n_i(k)$')
ax.loglog(wvn, e_vc_1[:Nx // 2] + e_vc_2[:Nx // 2], color='b', marker='D', markersize=2, linestyle='None',
          label=r'$n_c(k)$')
# ax.loglog(wvn, sum_1[:Nx // 2], color='g', marker='D', markersize=2, linestyle='None', label=r'$\Sigma n_\delta(k)$')

# ax.loglog(wvn, e_s[:Nx // 2], color='y', marker='D', markersize=2, linestyle='None', label=r'$n_s(k)$')
# ax.loglog(wvn, sum_1, color='g', marker='D', markersize=2, linestyle='None', label=r'$\Sigma n_\delta(k)$')

ax.loglog(wvn[100:], 4e4 * wvn[100:] ** (-2), 'k:', label=r'$k^{-2}$')
ax.loglog(wvn[5:80], 4e3 * wvn[5:80] ** (-4), 'k--', label=r'$k^{-4}$')
ax.legend()
plt.savefig('../../plots/twoComponent/paper/gamma06_spectra.pdf', bbox_inches='tight')
plt.show()
