import cupy as cp

"""Module file containing functions for solving the two-component GPEs using a symplectic method."""


def kinetic_evolution(wfn_k, time_step, wvn_x, wvn_y):
    wfn_k *= cp.exp(-0.25 * 1j * time_step * (wvn_x ** 2 + wvn_y ** 2))

    return wfn_k


def potential_evolution(wfn_1_old, wfn_2_old, time_step, inter_int, intra_int):
    wfn_1 = wfn_1_old * cp.exp(-1j * time_step * (intra_int * cp.abs(wfn_1_old) + inter_int * cp.abs(wfn_2_old)))
    wfn_2 = wfn_2_old * cp.exp(-1j * time_step * (intra_int * cp.abs(wfn_2_old) + inter_int * cp.abs(wfn_1_old)))

    return wfn_1, wfn_2
