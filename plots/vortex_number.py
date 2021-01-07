import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

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
maps_2000 = load_data('../data/vortexData/2CHQV_imp_Nvort=2000_VD.pkl')
maps_1500 = load_data('../data/vortexData/2CHQV_imp_Nvort=1500_VD.pkl')
maps_1000 = load_data('../data/vortexData/2CHQV_imp_Nvort=1000_VD.pkl')
maps_500 = load_data('../data/vortexData/2CHQV_imp_Nvort=500_VD.pkl')

# Empty vortex number lists:
vortexNum_2000 = []
vortexNum_1500 = []
vortexNum_1000 = []
vortexNum_500 = []

pickle_length = 0   # Counts number of elements in pickle file

# Loads in the maps and finds the total vortices in each Map and saves to a list
while True:
    try:
        current_map_2000 = next(maps_2000)
        current_map_1500 = next(maps_1500)
        current_map_1000 = next(maps_1000)
        current_map_500 = next(maps_500)

        vortexNum_2000.append(current_map_2000.total_vortices())
        vortexNum_1500.append(current_map_1500.total_vortices())
        vortexNum_1000.append(current_map_1000.total_vortices())
        vortexNum_500.append(current_map_500.total_vortices())

        pickle_length += 1
    except StopIteration:
        print('Exceeded length of pickle... Exiting loop.')
        break
print('Pickle length = {}'.format(pickle_length))

time_array = np.arange(100 * pickle_length, step=100)   # Time

# Set up plots:
fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(14, 12))
# ax.set_ylim(bottom=9e1, top=2e3)
ax[0, 0].set_ylabel(r'$N_\mathrm{vort}$')
ax[1, 0].set_ylabel(r'$N_\mathrm{vort}$')
ax[1, 0].set_xlabel(r'$t/\tau$')
ax[1, 1].set_xlabel(r'$t/\tau$')

# Loglog plot on each axis:
ax[0, 0].loglog(time_array[:], vortexNum_2000[:], markersize=2, linestyle='--', marker='o', color='r')
ax[0, 0].set_title(r'$N_{vort}=4000$')

ax[0, 1].loglog(time_array[:], vortexNum_1500[:], markersize=2, linestyle='--', marker='o', color='r')
ax[0, 1].set_title(r'$N_{vort}=3000$')

ax[1, 0].loglog(time_array[:], vortexNum_1000[:], markersize=2, linestyle='--', marker='o', color='r')
ax[1, 0].set_title(r'$N_{vort}=2000$')

ax[1, 1].loglog(time_array[:], vortexNum_500[:], markersize=2, linestyle='--', marker='o', color='r')
ax[1, 1].set_title(r'$N_{vort}=1000$')

# Plot scaling lines:
ax[0, 0].loglog(time_array[300:], 0.4e5 * (time_array[300:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[0, 0].loglog(time_array[50:300], 1.5e7 * (time_array[50:300] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[0, 0].loglog(time_array[300:], 1.7e4 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')
ax[0, 0].legend()

ax[0, 1].loglog(time_array[300:], 0.4e5 * (time_array[300:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[0, 1].loglog(time_array[50:300], 1.5e7 * (time_array[50:300] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[0, 1].loglog(time_array[300:], 1.7e4 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')

ax[1, 0].loglog(time_array[300:], 0.4e5 * (time_array[300:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[1, 0].loglog(time_array[50:300], 1.5e7 * (time_array[50:300] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[1, 0].loglog(time_array[300:], 1.7e4 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')

ax[1, 1].loglog(time_array[300:], 0.4e5 * (time_array[300:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[1, 1].loglog(time_array[50:300], 1.5e7 * (time_array[50:300] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[1, 1].loglog(time_array[300:], 1.7e4 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')

# plt.savefig('../../plots/twoComponent/HQV_imp_Nvort.png', bbox_inches='tight')
plt.show()
