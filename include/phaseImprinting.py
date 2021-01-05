import cupy as cp
import numpy as np


def get_phase(num_of_vort, pos, x_pts, y_pts, grid_x, grid_y, grid_len_x, grid_len_y):
    # Phase initialisation
    theta_tot = cp.empty((x_pts, y_pts))

    for k in range(num_of_vort // 2):
        theta_k = cp.zeros((x_pts, y_pts))

        y_m, y_p = pos[k], pos[num_of_vort + k]  # y-positions
        x_m, x_p = pos[num_of_vort // 2 + k], pos[3 * num_of_vort // 2 + k]  # x-positions

        # Scaling positional arguments
        Y_minus = 2 * cp.pi * (grid_y - y_m) / grid_len_y
        X_minus = 2 * cp.pi * (grid_x - x_m) / grid_len_x
        Y_plus = 2 * cp.pi * (grid_y - y_p) / grid_len_y
        X_plus = 2 * cp.pi * (grid_x - x_p) / grid_len_x
        x_plus = 2 * cp.pi * x_p / grid_len_x
        x_minus = 2 * cp.pi * x_m / grid_len_x

        heav_xp = cp.asarray(np.heaviside(cp.asnumpy(X_plus), 1.))
        heav_xm = cp.asarray(np.heaviside(cp.asnumpy(X_minus), 1.))

        for nn in cp.arange(-5, 5):
            theta_k += cp.arctan(cp.tanh((Y_minus + 2 * cp.pi * nn) / 2) * cp.tan((X_minus - cp.pi) / 2)) \
                                - cp.arctan(cp.tanh((Y_plus + 2 * cp.pi * nn) / 2) * cp.tan((X_plus - cp.pi) / 2)) \
                                + cp.pi * (heav_xp - heav_xm)

        theta_k -= (2 * cp.pi * grid_y / grid_len_y) * (x_plus - x_minus) / (2 * cp.pi)
        theta_tot += theta_k
    return theta_tot
