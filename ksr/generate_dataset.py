import numpy as np
import ks_dft, functionals, ext_potentials
import functools
import os
import sys


class LDA_atom_dataset():
    """Obtain dataset for LDA-calculated systems. Current support is for atoms.
    """

    def __init__(self, grids=None, selected_z=None, locations=None,
                 num_electrons=None):
        self.grids = grids
        self.selected_z = selected_z

        if grids is None and selected_z is None and locations is None:
            pass
        elif locations is None:
            self.locations = [[0]] * len(self.selected_z)
        else:
            self.locations = locations

        # output quantities
        self.num_electrons = num_electrons
        self.nuclear_charges = []
        self.total_energies = []
        self.densities = []
        self.external_potentials = []
        self.xc_energies = []

    def run_selected_ions(self):
        lda_xc = functionals.ExponentialLDAFunctional(grids=self.grids)
        for (z, center) in zip(self.selected_z, self.locations):
            v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=z,
                                      center=center)
            solver = ks_dft.Spinless_KS_Solver(self.grids, v_ext=v_ext,
                                               xc=lda_xc,
                                               num_electrons=self.num_electrons)
            solver.solve_self_consistent_density()
            print(
                'finished: (z, num_el) = (' + str(z) + ',' +
                str(self.num_electrons) + ')')

            self.nuclear_charges.append([z])
            self.total_energies.append(solver.E_tot)
            self.densities.append(solver.density)
            self.external_potentials.append(v_ext(self.grids))
            self.xc_energies.append(solver.E_x + solver.E_c)

    def save_dataset(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        np.save(os.path.join(out_dir, 'grids.npy'), self.grids)
        np.save(os.path.join(out_dir, 'locations.npy'), self.locations)
        np.save(os.path.join(out_dir, 'num_electrons.npy'), self.num_electrons)
        np.save(os.path.join(out_dir, 'nuclear_charges.npy'),
                self.nuclear_charges)
        np.save(os.path.join(out_dir, 'total_energies.npy'),
                self.total_energies)
        np.save(os.path.join(out_dir, 'densities.npy'),
                self.densities)
        np.save(os.path.join(out_dir, 'external_potentials.npy'),
                self.external_potentials)
        np.save(os.path.join(out_dir, 'xc_energies.npy'),
                self.xc_energies)


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # ions are identified by: atomic number Z, number of electrons
    selected_z = [1, 2, 3, 4]
    num_electrons = 1
    out_dir = 'num_electrons_'+str(num_electrons)
    out_dir = os.path.join('atoms', out_dir)

    dataset = LDA_atom_dataset(grids, selected_z, num_electrons=num_electrons)
    dataset.run_selected_ions()
    dataset.save_dataset(out_dir=out_dir)
