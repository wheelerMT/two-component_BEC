import h5py
import numpy as np

"""File that calculates certain correlation functions and lengths from a given dataset and saves to a new dataset."""

# Import data:
filename = input('Enter filename: ')
data_path = '../data/{}.hdf5'.format(filename)
data_file = h5py.File(data_path, 'r')

# Loading grid array data:
x, y = data_file['grid/x'][:], data_file['grid/y'][:]
X, Y = np.meshgrid(x[:], y[:])
Nx, Ny = x[:].size, y[:].size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)

# Three component wavefunction
psi_1 = data_file['wavefunction/psi_1']
psi_2 = data_file['wavefunction/psi_2']
n_0 = abs(psi_1[:, :, 0]) ** 2 + abs(psi_2[:, :, 0]) ** 2  # Total background density

num_of_frames = psi_1.shape[-1]
frame_start = 0

# Calculate phase OP:
integral_phase = np.empty((Nx, Ny, num_of_frames - frame_start))
for i in range(frame_start, num_of_frames):
    alpha_perp = np.fft.fft2(-2 * psi_1[:, :, i] * psi_2[:, :, i])
    integral_phase[:, :, i - frame_start] = np.fft.fftshift(1 / (n_0 * Nx) ** 2 * np.fft.ifft2(alpha_perp * np.conj(alpha_perp))).real
    integral_phase[:, :, i] = np.where(integral_phase[:, :, i] < 0, 0, integral_phase[:, :, i])

# Calculate nematic OP:
integral_nematic = np.empty((Nx, Ny, num_of_frames))
for i in range(frame_start, num_of_frames):
    q_xx_k = np.fft.fft2(np.real(np.conj(psi_1[:, :, i]) * psi_2[:, :, i]))
    q_xy_k = np.fft.fft2(np.imag(np.conj(psi_1[:, :, i]) * psi_2[:, :, i]))
    integral_nematic[:, :, i - frame_start] = np.fft.fftshift(4 / (n_0 * Nx) ** 2 * np.fft.ifft2(q_xx_k * np.conj(q_xx_k) + q_xy_k * np.conj(q_xy_k))).real
    integral_nematic[:, :, i] = np.where(integral_nematic[:, :, i] < 0, 0, integral_nematic[:, :, i])

data_file.close()

box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)
centerx = Nx // 2
centery = Ny // 2

# Generate data file:
corr_file = '{}_corr'.format(filename)
corr_data = h5py.File('../data/{}.hdf5'.format(corr_file), 'w')
g_theta = corr_data.create_dataset('g_theta', (box_radius, num_of_frames - frame_start))
g_phi = corr_data.create_dataset('g_phi', (box_radius, num_of_frames - frame_start))

# Sum over the angle:
print('Calculating angular average...')
for frame in range(num_of_frames - frame_start):
    angle_sum = np.zeros(box_radius, )
    nc = np.zeros(box_radius, )
    print('On frame {}.'.format(frame + 1))
    for i in range(Nx):
        for j in range(Ny):
            r = int(np.round(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2)))
            nc[r] += 1
            angle_sum[r] += integral_phase[i, j, frame]
    angle_sum /= nc
    g_theta[:, frame] = angle_sum

# Sum over the angle:
print('Calculating angular average...')
for frame in range(num_of_frames - frame_start):
    angle_sum = np.zeros(box_radius, )
    nc = np.zeros(box_radius, )
    print('On frame {}.'.format(frame + 1))
    for i in range(Nx):
        for j in range(Ny):
            r = int(np.round(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2)))
            nc[r] += 1
            angle_sum[r] += integral_nematic[i, j, frame]
    angle_sum /= nc
    g_phi[:, frame] = angle_sum

corr_data.close()
