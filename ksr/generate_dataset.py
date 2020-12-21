import numpy as np
import ks_dft, functionals, ext_potentials
import functools


class LDA_atom_dataset():
    def __init__(self, grids, selected_ions, locations):
        self.grids = grids
        self.selected_ions = selected_ions
        self.locations = locations

        # output quantities
        self.num_electrons = []
        self.locations = []


    def run_selected_ions(self):

        lda_xc = functionals.ExponentialLDAFunctional(grids=self.grids)
        for (z, num_el) in self.selected_ions:
            v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=z)
            solver = ks_dft.KS_Solver(self.grids, v_ext=v_ext, xc=lda_xc,
                                      num_electrons=num_el)
            solver.solve_self_consistent_density()

if __name__ == '__main__':
    selected_ions = [(2,2), (3,2)]

