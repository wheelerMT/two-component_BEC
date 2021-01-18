import cupy as cp
import numpy as np


def get_phase(num_of_vort, pos, x_pts, y_pts, grid_x, grid_y, grid_len_x, grid_len_y):

    """ Gets phase distribution of N dipoles."""

    # Phase initialisation
    theta_tot = cp.empty((x_pts, y_pts))

    # Scale pts:
    x_tilde = 2 * cp.pi * ((grid_x - grid_x.min()) / grid_len_x)
    y_tilde = 2 * cp.pi * ((grid_y - grid_y.min()) / grid_len_y)

    switch = False

    for i in range(num_of_vort // 2):
        theta_k = cp.zeros((x_pts, y_pts))

        if i % 24 == 0 and i > 0:
            switch ^= True

        if switch:
            x_p, y_p = next(pos)
            x_m, y_m = next(pos)

        else:
            x_m, y_m = next(pos)
            x_p, y_p = next(pos)

        # Scaling vortex positions:
        x_m_tilde = 2 * cp.pi * ((x_m - grid_x.min()) / grid_len_x)
        y_m_tilde = 2 * cp.pi * ((y_m - grid_y.min()) / grid_len_y)
        x_p_tilde = 2 * cp.pi * ((x_p - grid_x.min()) / grid_len_x)
        y_p_tilde = 2 * cp.pi * ((y_p - grid_y.min()) / grid_len_y)

        # Aux variables
        Y_minus = y_tilde - y_m_tilde
        X_minus = x_tilde - x_m_tilde
        Y_plus = y_tilde - y_p_tilde
        X_plus = x_tilde - x_p_tilde

        heav_xp = cp.asarray(np.heaviside(cp.asnumpy(X_plus), 1.))
        heav_xm = cp.asarray(np.heaviside(cp.asnumpy(X_minus), 1.))

        for nn in cp.arange(-5, 6):
            theta_k += cp.arctan(cp.tanh((Y_minus + 2 * cp.pi * nn) / 2) * cp.tan((X_minus - cp.pi) / 2)) \
                                - cp.arctan(cp.tanh((Y_plus + 2 * cp.pi * nn) / 2) * cp.tan((X_plus - cp.pi) / 2)) \
                                + cp.pi * (heav_xp - heav_xm)

        theta_k -= y_tilde * (x_p_tilde - x_m_tilde) / (2 * cp.pi)
        theta_tot += theta_k

    return theta_tot
