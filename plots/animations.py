import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fft2, ifft2


def spectral_derivative(array, wvn):
    return ifft2((1j * wvn * (fft2(array))))


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
Nx, Ny = x[:].size, y[:].size
X, Y = np.meshgrid(x[:], y[:])

# k-space arrays and meshgrid:
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)  # K-space spacing
kx = np.arange(-Nx // 2, Nx // 2) * dkx
ky = np.arange(-Ny // 2, Ny // 2) * dky
Kx, Ky = np.meshgrid(kx, ky)  # K-space meshgrid
Kx, Ky = np.fft.ifftshift(Kx), np.fft.ifftshift(Ky)


# Loading time variables:
Nt, dt, Nframe = np.array(data_file['time/Nt']), np.array(data_file['time/dt']), np.array(data_file['time/Nframe'])
num_of_frames = psi_1.shape[-1]

# ------------------------------------------------------------------------------------------------------------------
# Set up initial figure
# ------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
ax[0, 0].set_ylabel(r'$y / a_s$')
ax[1, 0].set_ylabel(r'$y / a_s$')
ax[0, 0].set_title(r'$|\psi_1|^2$')
ax[0, 1].set_title(r'$|\psi_2|^2$')
ax[1, 0].set_title(r'$\nabla \times n_1\vec{v}_1$')
ax[1, 1].set_title(r'$\nabla \times n_2\vec{v}_2$')
for axis in ax[1, :]:
    axis.set_xlabel(r'$x / a_s$')

# Initial frame plot:
densPlus_plot = ax[0, 0].pcolormesh(X, Y, abs(psi_1[:, :, 0]) ** 2, vmin=0, vmax=3500, shading='auto', cmap='gnuplot')
densMinus_plot = ax[0, 1].pcolormesh(X, Y, abs(psi_2[:, :, 0]) ** 2, vmin=0, vmax=3500, shading='auto', cmap='gnuplot')

# Set up color bar:
dens_cbar = plt.colorbar(densMinus_plot, ax=ax[0, 1], fraction=0.044, pad=0.03)

# Calculate the pseudo-vorticity:
dpsi_x_1 = spectral_derivative(psi_1[:, :, 0], Kx)
dpsi_y_1 = spectral_derivative(psi_1[:, :, 0], Ky)
dpsi_x_2 = spectral_derivative(psi_2[:, :, 0], Kx)
dpsi_y_2 = spectral_derivative(psi_2[:, :, 0], Ky)

pseudo_vort_1 = 2 * (np.conj(dpsi_x_1) * dpsi_y_1 - dpsi_x_1 * np.conj(dpsi_y_1)) / (2 * 1j)
pseudo_vort_2 = 2 * (np.conj(dpsi_x_2) * dpsi_y_2 - dpsi_x_2 * np.conj(dpsi_y_2)) / (2 * 1j)

pseudoVortPlus_plot = ax[1, 0].pcolormesh(X, Y, pseudo_vort_1.real, vmin=-500, vmax=500, shading='auto', cmap='jet')
pseudoVortMinus_plot = ax[1, 1].pcolormesh(X, Y, pseudo_vort_2.real, vmin=-500, vmax=500, shading='auto', cmap='jet')

vort_cbar = plt.colorbar(pseudoVortMinus_plot, ax=ax[1, 1], fraction=0.044, pad=0.03)
plt.tight_layout()

cont = [densPlus_plot, densMinus_plot, pseudoVortPlus_plot, pseudoVortMinus_plot]


def animate(i):
    """Animation function for plots."""

    # Calculate the pseudo-vorticity:
    dpsi_x_1 = spectral_derivative(psi_1[:, :, i], Kx)
    dpsi_y_1 = spectral_derivative(psi_1[:, :, i], Ky)
    dpsi_x_2 = spectral_derivative(psi_2[:, :, i], Kx)
    dpsi_y_2 = spectral_derivative(psi_2[:, :, i], Ky)

    pseudo_vort_1 = (2 * (np.conj(dpsi_x_1) * dpsi_y_1 - dpsi_x_1 * np.conj(dpsi_y_1)) / (2 * 1j)).real
    pseudo_vort_2 = (2 * (np.conj(dpsi_x_2) * dpsi_y_2 - dpsi_x_2 * np.conj(dpsi_y_2)) / (2 * 1j)).real

    densPlus_plot.set_array((abs(psi_1[:, :, i]) ** 2).ravel())
    densMinus_plot.set_array((abs(psi_1[:, :, i]) ** 2).ravel())

    pseudoVortPlus_plot.set_array(pseudo_vort_1.ravel())
    pseudoVortMinus_plot.set_array(pseudo_vort_2.ravel())

    cont = [densPlus_plot, densMinus_plot, pseudoVortPlus_plot, pseudoVortMinus_plot]

    print('On density iteration %i' % (i + 1))
    plt.suptitle(r'$\tau$ = %2f' % (Nframe * dt * i))
    return cont


# Calls the animation function and saves the result
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames, blit=False, repeat=False)
anim.save('../../plots/twoComponent/{}.mp4'.format(filename),
          writer=animation.FFMpegWriter(fps=60, codec="libx264", extra_args=['-pix_fmt', 'yuv420p']))
print('Density video saved successfully.')
