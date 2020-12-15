import numpy as np


def get_phase(num_of_vort, pos, x_pts, y_pts, grid_x, grid_y, grid_len_x, grid_len_y):
    # Phase initialisation
    theta_k = np.zeros((num_of_vort, x_pts, y_pts))
    theta_tot = np.zeros((x_pts, y_pts))

    for k in range(num_of_vort // 2):
        y_m, y_p = pos[k], pos[num_of_vort + k]  # y-positions
        x_m, x_p = pos[num_of_vort // 2 + k], pos[3 * num_of_vort // 2 + k]  # x-positions

        # Scaling positional arguments
        Y_minus = 2 * np.pi * (grid_y - y_m) / grid_len_y
        X_minus = 2 * np.pi * (grid_x - x_m) / grid_len_x
        Y_plus = 2 * np.pi * (grid_y - y_p) / grid_len_y
        X_plus = 2 * np.pi * (grid_x - x_p) / grid_len_x
        x_plus = 2 * np.pi * x_p / grid_len_x
        x_minus = 2 * np.pi * x_m / grid_len_x

        heav_xp = np.heaviside(X_plus, 1.)
        heav_xm = np.heaviside(X_minus, 1.)

        for nn in np.arange(-5, 5):
            theta_k[k, :, :] += np.arctan(np.tanh((Y_minus + 2 * np.pi * nn) / 2) * np.tan((X_minus - np.pi) / 2)) \
                                - np.arctan(np.tanh((Y_plus + 2 * np.pi * nn) / 2) * np.tan((X_plus - np.pi) / 2)) \
                                + np.pi * (heav_xp - heav_xm)

        theta_k[k, :, :] -= (2 * np.pi * grid_y / grid_len_y) * (x_plus - x_minus) / (2 * np.pi)
        theta_tot += theta_k[k, :, :]
    return theta_tot
