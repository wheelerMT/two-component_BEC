import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

""" File that loads in data from a vortex data pkl file and plots the corresponding vortex number."""


def load_data(Filename):
    with open(Filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# Load in vortex map pickled data:
filename = 'HQV_grid_gamma=075'
maps = load_data('../data/vortexData/{}_VD.pkl'.format(filename))
pickle_length = 0   # Counts number of elements in pickle file

# Empty vortex number list:
vortex_num = []

# Loads in the maps and finds the total vortices in each Map and saves to a list
while True:
    try:
        current_map = next(maps)
        vortex_num.append(current_map.total_vortices())

        pickle_length += 1
    except StopIteration:
        print('Exceeded length of pickle... Exiting loop.')
        break
print('Pickle length = {}'.format(pickle_length))

time_array = np.arange(100 * pickle_length, step=100)   # Time

# Set up plots:
fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].set_ylabel(r'$N_\mathrm{vort}$')
ax[1].set_ylabel(r'$\ell_d$')
ax[0].set_xlabel(r'$t/\tau$')
ax[1].set_xlabel(r'$t/\tau$')

ax[0].loglog(time_array[:], vortex_num[:], 'rD')
ax[0].loglog(time_array[300:], 0.2e5 * (time_array[300:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[0].loglog(time_array[20:250], 0.3e7 * (time_array[20:250] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[0].loglog(time_array[300:], 0.5e4 * (time_array[300:] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')
ax[0].legend(fontsize=16)

ax[1].loglog(time_array[:], 1/np.sqrt(np.array(vortex_num[:])), 'rD')
ax[1].loglog(time_array[10:100], 0.9e-3 * (time_array[10:100] ** (1./2)), 'k--', label=r'$t^{\frac{1}{2}}$')
ax[1].loglog(time_array[300:], 0.8e-2 * (time_array[300:] ** (1./4)), 'k-', label=r'$t^{\frac{1}{4}}$')
ax[1].loglog(time_array[1:10], 0.55e-2 * (time_array[1:10] ** (1./5)), 'k:', label=r'$t^{\frac{1}{5}}$')
ax[1].legend(fontsize=16)

# plt.savefig('../../plots/twoComponent/{}_Nvort.png'.format(filename), bbox_inches='tight')
plt.show()
