import numpy as np
import uuid


class Vortex:
    """ Vortex class to categorize vortices within a BEC."""
    def __init__(self, position, winding, component, v_type=None):
        self.x, self.y = position
        self.winding = winding
        self.v_type = v_type  # String: type of vortex (i.e. SQV or HQV)
        self.uid = '{}_{}'.format(v_type, uuid.uuid1())  # Unique identifier string
        self.isTracked = True   # Tracking argument for vortex
        self.component = component  # Which component of the wavefunction the vortex is in

    def get_coords(self):
        return self.x, self.y

    def get_uid(self):
        return self.uid

    def get_v_type(self):
        return self.v_type

    def get_distance(self, vortex):  # Calculate distance between two vortices:
        return np.sqrt((self.x - vortex.x) ** 2 + (self.y - vortex.y) ** 2)

    def update_type(self, vortex_type):
        self.v_type = vortex_type

    def update_uid(self):
        self.uid = '{}_{}'.format(self.v_type, self.uid)

    def update_coords(self, pos_x, pos_y):
        self.x, self.y = pos_x, pos_y


class VortexMap:
    """Map that keeps track of all vortices within a condensate."""

    def __init__(self):
        self.vortices_unid = []  # Unidentified vortices
        self.vortices_sqv = []
        self.vortices_hqv = []

    def add_vortex(self, vortex):
        # * Adds a vortex to the unidentified pool of the vortexMap
        if vortex.v_type == 'SQV':
            self.vortices_sqv.append(vortex)
        elif vortex.v_type == 'HQV':
            self.vortices_hqv.append(vortex)
        else:
            self.vortices_unid.append(vortex)

    def sort_vortices(self, vortex):
        # * Function that sorts all identified vortices into their respective pools
        if vortex.v_type == 'SQV':
            self.vortices_sqv.append(vortex)
        if vortex.v_type == 'HQV':
            self.vortices_hqv.append(vortex)

    def num_of_vortices(self):
        print('There are {} SQVs and {} HQVs in the system.'.format(len(self.vortices_sqv), len(self.vortices_hqv)))
        return len(self.vortices_sqv) + len(self.vortices_hqv)

    def identify_vortices(self, threshold):
        # * Finds SQVs by finding overlapping vortices in the components
        # * Threshold determines the maximum distance between to cores to be classed as a SQV

        vortices_1 = [vortex for vortex in self.vortices_unid if vortex.component == '1']
        vortices_2 = [vortex for vortex in self.vortices_unid if vortex.component == '2']

        for vortex_1 in vortices_1:
            for vortex_2 in vortices_2:
                if abs(vortex_1.x - vortex_2.x) < threshold:
                    if abs(vortex_1.y - vortex_2.y) < threshold:
                        # * If this evaluates to true, the two vortices are within the threshold
                        # * Firstly, get the average of the positions of the two overlapping vortices
                        sqv_pos = (vortex_1.x + vortex_2.x) / 2, (vortex_1.y + vortex_2.y) / 2

                        # * Generate new SQV vortex that gets added to the SQV pool
                        self.add_vortex(Vortex(sqv_pos, vortex_1.winding, component='both', v_type='SQV'))

                        # * Remove the corresponding vortex_plus and vortex_minus from the unid pool
                        if vortex_1 in self.vortices_unid:
                            self.vortices_unid.remove(vortex_1)
                        if vortex_2 in self.vortices_unid:
                            self.vortices_unid.remove(vortex_2)
                        break

        # * Determines HQVs by setting all remaining unidentified vortices to HQVs
        for vortex in self.vortices_unid:
            vortex.update_type('HQV')
            vortex.update_uid()
            self.vortices_hqv.append(vortex)

        self.vortices_unid = []  # Empties unid list
