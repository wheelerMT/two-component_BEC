import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 16})

""" File that loads in data from a vortex data pkl file and plots the corresponding vortex number."""


def load_data(Filename):
    with open(Filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# Load in vortex map pickled data:
filename = 'HQV_grid_gamma=01'  # input('Enter filename: ')
maps = load_data('../data/vortexData/{}_VD.pkl'.format(filename))
pickle_length = 0   # Counts number of elements in pickle file

# Empty vortex number list:
psi_1_vortex = []
psi_2_vortex = []

# Loads in the maps and finds the total vortices in each Map and saves to a list
while True:
    try:
        current_map = next(maps)
        psi_1_vortex.append(current_map.hqv_number('1'))
        psi_2_vortex.append(current_map.hqv_number('2'))

        pickle_length += 1
    except StopIteration:
        print('Exceeded length of pickle... Exiting loop.')
        break
print('Pickle length = {}'.format(pickle_length))

time_array = np.arange(200 * pickle_length, step=200)   # Time
psi_1_vortex = np.array(psi_1_vortex)
psi_2_vortex = np.array(psi_2_vortex)

# Set up plots:
fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].set_ylabel(r'$N_\mathrm{vort}$')
ax[1].set_ylabel(r'$\ell_d$')
ax[0].set_xlabel(r'$t/\tau$')
ax[1].set_xlabel(r'$t/\tau$')

ax[0].loglog(time_array[:], psi_1_vortex[:], 'rD', markersize=2, label=r'$\psi_1$')
ax[0].loglog(time_array[:], psi_2_vortex[:], 'bD', markersize=2, label=r'$\psi_2$')
ax[0].loglog(time_array[:], psi_1_vortex[:] + psi_2_vortex[:], 'kD', markersize=2, label=r'Total')
ax[0].loglog(time_array[50:], 2e4 * (time_array[50:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[0].loglog(time_array[3:20], 7e5 * (time_array[3:20] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[0].loglog(time_array[300:], 5e3 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')
ax[0].legend(fontsize=16)

ax[1].loglog(time_array[:], 1/np.sqrt(psi_1_vortex[:]), 'rD', markersize=2, label=r'$\psi_1$')
ax[1].loglog(time_array[:], 1/np.sqrt(psi_2_vortex[:]), 'bD', markersize=2, label=r'$\psi_2$')
ax[1].loglog(time_array[:], 1/np.sqrt(psi_1_vortex[:] + psi_2_vortex[:]), 'kD', markersize=2, label=r'Total')
ax[1].loglog(time_array[50:], 0.7e-2 * (time_array[50:] ** (1./4)), 'k-', label=r'$t^{\frac{1}{4}}$')
ax[1].loglog(time_array[3:20], 1e-3 * (time_array[3:20] ** (1./2)), 'k--', label=r'$t^{\frac{1}{2}}$')
ax[1].loglog(time_array[300:], 1.5e-2 * (time_array[300:] ** (1./5)), 'k:', label=r'$t^{\frac{1}{5}}$')
ax[1].legend(fontsize=16)

# ax[2].loglog(time_array[:], abs(np.array(psi_1_vortex[:]) - np.array(psi_2_vortex)), 'rD', label=r'$\psi_1$')
# ax[2].set_ylabel(r'$|N_1 - N_2|$')
# ax[2].set_xlabel(r'$t/\tau$')

plt.savefig('../../plots/twoComponent/{}_Nvort.png'.format(filename), bbox_inches='tight')
plt.show()
