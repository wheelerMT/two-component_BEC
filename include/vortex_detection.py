import numpy as np
import vortex.vortex as vortex
import time


def detect(vortexMap, pos_x, pos_y, wfn, x_grid, y_grid, component):
    """A plaquette detection algorithm that finds areas of 2pi winding and then performs a least squares
    fit in order to obtain exact position of the vortex core."""

    t1 = time.time()
    counter = 0  # Used for counting vortices

    # Unwrap phase to avoid discontinuities:
    phase_x = np.unwrap(np.angle(wfn), axis=0)
    phase_y = np.unwrap(np.angle(wfn), axis=1)

    # Sum phase difference over plaquettes:
    for ii, jj in zip(pos_x, pos_y):
        # Ensures algorithm works at edge of box:
        if ii == len(x_grid) - 1:
            ii = -1
        if jj == len(y_grid) - 1:
            jj = -1

        phase_sum = 0
        phase_sum += phase_x[ii, jj + 1] - phase_x[ii, jj]
        phase_sum += phase_y[ii + 1, jj + 1] - phase_y[ii, jj + 1]
        phase_sum += phase_x[ii + 1, jj] - phase_x[ii + 1, jj + 1]
        phase_sum += phase_y[ii, jj] - phase_y[ii + 1, jj]

        # If sum of phase difference is 2pi or -2pi, create vortex object::
        if np.round(abs(phase_sum), 4) == np.round(2 * np.pi, 4):
            if phase_sum > 0:  # Anti-Vortex
                vortexMap.add_vortex(vortex.Vortex(refine_positions([ii, jj], wfn, x_grid, y_grid), -1, component))
            elif phase_sum < 0:  # Vortex
                vortexMap.add_vortex(vortex.Vortex(refine_positions([ii, jj], wfn, x_grid, y_grid), 1, component))
            counter += 1

    print('Found {} vortices in {:.2f}s!'.format(counter, time.time() - t1))


def refine_positions(position, wfn, x_grid, y_grid):
    """ Perform a least squares fitting to correct the vortex positions."""

    x_pos, y_pos = position

    x_update = (-wfn[x_pos, y_pos] + wfn[x_pos, y_pos + 1] - wfn[x_pos + 1, y_pos] + wfn[x_pos + 1, y_pos + 1]) / 2
    y_update = (-wfn[x_pos, y_pos] - wfn[x_pos, y_pos + 1] + wfn[x_pos + 1, y_pos] + wfn[x_pos + 1, y_pos + 1]) / 2
    c_update = (3 * wfn[x_pos, y_pos] + wfn[x_pos, y_pos + 1] + wfn[x_pos + 1, y_pos] - wfn[
        x_pos + 1, y_pos + 1]) / 4

    Rx, Ry = x_update.real, y_update.real
    Ix, Iy = x_update.imag, y_update.imag
    Rc, Ic = c_update.real, c_update.imag

    det = 1 / (Rx * Iy - Ry * Ix)
    delta_x = det * (Iy * Rc - Ry * Ic)
    delta_y = det * (-Ix * Rc + Rx * Ic)

    x_v = x_pos - delta_x
    y_v = y_pos - delta_y

    if x_v > len(x_grid) - 1:
        x_v -= (len(x_grid) - 1)
    if y_v > len(y_grid) - 1:
        y_v -= (len(y_grid) - 1)
    if x_v < 0:
        x_v += (len(x_grid) - 1)
    if y_v < 0:
        y_v += (len(y_grid) - 1)

    # Return x and y positions:
    return (y_v - len(y_grid) // 2) * (y_grid[1] - y_grid[0]), (x_v - len(x_grid) // 2) * (x_grid[1] - x_grid[0])


def remove_invalid_positions(vortex_pos, antivortex_pos, r):
    """ Function that removes oppositely signed vortices that are too close together, typically these are sometimes
    detected by the algorithm when the phase field is broken near an existing vortex."""

    removable_vort = []  # List of removable vortices
    removable_antivort = []  # List of removable anti-vortices

    for vtx in vortex_pos:
        position_found = False

        for anti_vtx in antivortex_pos:
            if (vtx[0] - anti_vtx[0]) ** 2 + (vtx[1] - anti_vtx[1]) ** 2 < r:
                if anti_vtx not in removable_antivort:
                    removable_antivort.append(anti_vtx)
                    position_found = True

        if position_found and vtx not in removable_vort:
            removable_vort.append(vtx)

    for vtx in removable_vort:
        if vtx in vortex_pos:
            del vortex_pos[vortex_pos.index(vtx)]

    for anti_vtx in removable_antivort:
        if anti_vtx in antivortex_pos:
            del antivortex_pos[antivortex_pos.index(anti_vtx)]

    return vortex_pos, antivortex_pos
