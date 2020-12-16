import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
Nt, dt, Nframe = np.array(data_file['time/Nt']), np.array(data_file['time/dt']), np.array(data_file['time/Nframe'])
num_of_frames = psi_1.shape[-1]

# ------------------------------------------------------------------------------------------------------------------
# Set up initial figure
# ------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
ax[0].set_ylabel(r'$y / \xi$')
ax[0].set_title(r'$|\psi_1|^2$')
ax[1].set_title(r'$|\psi_2|^2$')
for axis in ax:
    axis.set_xlabel(r'$x / \xi$')

cvals_dens = np.linspace(0, 1, 25, endpoint=True)

# Initial frame plot:
densPlus_plot = ax[0].contourf(X, Y, abs(psi_1[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
densMinus_plot = ax[1].contourf(X, Y, abs(psi_2[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
cont = [densPlus_plot, densMinus_plot]

# Set up color bar:
dens_cbar = plt.colorbar(densMinus_plot, ax=ax[1], fraction=0.044, pad=0.03)

# Set axes to have equal aspect ratio:
for axis in ax:
    axis.set_aspect('equal')


def animate(i):
    """Animation function for plots."""
    global cont
    for contour in cont:
        for c in contour.collections:
            c.remove()

    ax[0].contourf(X, Y, abs(psi_1[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')
    ax[1].contourf(X, Y, abs(psi_2[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')

    cont = [ax[0], ax[1]]
    print('On density iteration %i' % (i + 1))
    plt.suptitle(r'$\tau$ = %2f' % (Nframe * dt * i), y=0.7)
    return cont


# Calls the animation function and saves the result
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames, repeat=False)
anim.save('../../plots/twoComponent/{}.mp4'.format(filename), dpi=200,
          writer=animation.FFMpegWriter(fps=60, codec="libx264", extra_args=['-pix_fmt', 'yuv420p']))
print('Density video saved successfully.')
