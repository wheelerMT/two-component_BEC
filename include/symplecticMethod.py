import cupy as cp

"""Module file containing functions for solving the two-component GPEs using a symplectic method."""


def kinetic_evolution(wfn_k, time_step, wvn_x, wvn_y):
    wfn_k *= cp.exp(-0.25 * 1j * time_step * (wvn_x ** 2 + wvn_y ** 2))


def potential_evolution(wfn_1_old, wfn_2_old, time_step, intra_1, intra_2, inter_int, cp_1, cp_2):
    wfn_1 = wfn_1_old * cp.exp(-1j * time_step * (intra_1 * cp.abs(wfn_1_old) ** 2 + inter_int * cp.abs(wfn_2_old) ** 2
                               - cp_1))
    wfn_2 = wfn_2_old * cp.exp(-1j * time_step * (intra_2 * cp.abs(wfn_2_old) ** 2 + inter_int * cp.abs(wfn_1_old) ** 2
                               - cp_2))

    return wfn_1, wfn_2
