import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 16})

""" File that loads in data from 4 datasets with differing initial amount of vortices and plots the vortex number 
    as a function of time."""


def load_data(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# Load in vortex map pickled data:
maps_08 = load_data('../data/vortexData/HQV_grid_gamma=08_VD.pkl')
maps_06 = load_data('../data/vortexData/HQV_grid_gamma=06_VD.pkl')
maps_03 = load_data('../data/vortexData/HQV_grid_gamma=03_VD.pkl')
maps_01 = load_data('../data/vortexData/HQV_grid_gamma=01_VD.pkl')

# Empty vortex number lists:
vortexNum_08 = []
vortexNum_06 = []
vortexNum_03 = []
vortexNum_01 = []

pickle_length = 0   # Counts number of elements in pickle file

# Loads in the maps and finds the total vortices in each Map and saves to a list
while True:
    try:
        current_map_08 = next(maps_08)
        current_map_06 = next(maps_06)
        current_map_03 = next(maps_03)
        current_map_01 = next(maps_01)

        vortexNum_08.append(current_map_08.total_vortices())
        vortexNum_06.append(current_map_06.total_vortices())
        vortexNum_03.append(current_map_03.total_vortices())
        vortexNum_01.append(current_map_01.total_vortices())

        pickle_length += 1
    except StopIteration:
        print('Exceeded length of pickle... Exiting loop.')
        break
print('Pickle length = {}'.format(pickle_length))

time_array = np.arange(200 * pickle_length, step=200)   # Time

# Set up plots:
fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(14, 12))
# ax.set_ylim(bottom=9e1, top=2e3)
ax[0, 0].set_ylabel(r'$N_\mathrm{vort}$')
ax[1, 0].set_ylabel(r'$N_\mathrm{vort}$')
ax[1, 0].set_xlabel(r'$t/\tau$')
ax[1, 1].set_xlabel(r'$t/\tau$')

# Loglog plot on each axis:
ax[0, 0].loglog(time_array[:], vortexNum_08[:], markersize=2, linestyle='--', marker='o', color='r')
ax[0, 0].set_title(r'$\gamma=0.8$')

ax[0, 1].loglog(time_array[:], vortexNum_06[:], markersize=2, linestyle='--', marker='o', color='r')
ax[0, 1].set_title(r'$\gamma=0.6$')

ax[1, 0].loglog(time_array[:], vortexNum_03[:], markersize=2, linestyle='--', marker='o', color='r')
ax[1, 0].set_title(r'$\gamma=0.3$')

ax[1, 1].loglog(time_array[:], vortexNum_01[:], markersize=2, linestyle='--', marker='o', color='r')
ax[1, 1].set_title(r'$\gamma=0.1$')

# Plot scaling lines:
ax[0, 0].loglog(time_array[50:], 2e4 * (time_array[50:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[0, 0].loglog(time_array[3:20], 7e5 * (time_array[3:20] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[0, 0].loglog(time_array[300:], 5e3 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')
ax[0, 0].legend()

ax[0, 1].loglog(time_array[50:], 2e4 * (time_array[50:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[0, 1].loglog(time_array[3:20], 7e5 * (time_array[3:20] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[0, 1].loglog(time_array[300:], 5e3 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')

ax[1, 0].loglog(time_array[50:], 5e4 * (time_array[50:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[1, 0].loglog(time_array[3:20], 7e5 * (time_array[3:20] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[1, 0].loglog(time_array[300:], 2e4 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')

ax[1, 1].loglog(time_array[50:], 6e4 * (time_array[50:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[1, 1].loglog(time_array[3:20], 7e5 * (time_array[3:20] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[1, 1].loglog(time_array[300:], 3e4 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')

# plt.savefig('../../plots/twoComponent/HQV_grid_Nvort.png', bbox_inches='tight')
plt.show()
