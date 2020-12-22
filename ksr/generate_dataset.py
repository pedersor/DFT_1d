import numpy as np
import ks_dft, functionals, ext_potentials
import functools
import os
import sys


class LDA_atom_dataset():
    """Obtain dataset for LDA-calculated systems. Current support is for atoms.
    """

    def __init__(self, grids=None, selected_ions=None, locations=None):
        self.grids = grids
        self.selected_ions = selected_ions

        if grids is None and selected_ions is None and locations is None:
            pass
        elif locations is None:
            self.locations = [[0]] * len(self.selected_ions)
        else:
            self.locations = locations

        # output quantities
        self.num_electrons = []
        self.nuclear_charges = []
        self.total_energies = []
        self.densities = []
        self.external_potentials = []

    def run_selected_ions(self):
        lda_xc = functionals.ExponentialLDAFunctional(grids=self.grids)
        for ((z, num_el), center) in zip(self.selected_ions, self.locations):
            v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=z,
                                      center=center)
            solver = ks_dft.KS_Solver(self.grids, v_ext=v_ext, xc=lda_xc,
                                      num_electrons=num_el)
            solver.solve_self_consistent_density()
            print(
                'finished: (z, num_el) = (' + str(z) + ',' + str(num_el) + ')')

            self.nuclear_charges.append([z])
            self.total_energies.append(solver.E_tot)
            self.densities.append(solver.density)
            self.external_potentials.append(v_ext(self.grids))

    def save_dataset(self, out_dir):
        np.save(os.path.join(out_dir, 'grids.npy'), self.grids)
        np.save(os.path.join(out_dir, 'locations.npy'), self.locations)

        # TODO: support num_electrons array...
        # just two electrons for now (need to update KSR code)
        self.num_electrons = 2
        np.save(os.path.join(out_dir, 'num_electrons.npy'), self.num_electrons)

        np.save(os.path.join(out_dir, 'nuclear_charges.npy'),
                self.nuclear_charges)
        np.save(os.path.join(out_dir, 'total_energies.npy'),
                self.total_energies)
        np.save(os.path.join(out_dir, 'densities.npy'),
                self.densities)
        np.save(os.path.join(out_dir, 'external_potentials.npy'),
                self.external_potentials)

    def open_dataset(self, in_dir):
        self.grids = np.load(os.path.join(in_dir, 'grids.npy'))
        self.external_potentials = np.load(
            os.path.join(in_dir, 'external_potentials.npy'))
        self.locations = np.load(os.path.join(in_dir, 'locations.npy'))
        self.nuclear_charges = np.load(
            os.path.join(in_dir, 'nuclear_charges.npy'))
        self.num_electrons = np.load(os.path.join(in_dir, 'num_electrons.npy'))
        self.densities = np.load(os.path.join(in_dir, 'densities.npy'))
        self.total_energies = np.load(
            os.path.join(in_dir, 'total_energies.npy'))


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # ions are identified by: (atomic number Z, total number of electrons).
    selected_ions = [(2, 2), (3, 2), (4, 2)]

    dataset = LDA_atom_dataset(grids, selected_ions)
    dataset.run_selected_ions()
    dataset.save_dataset(out_dir='atoms/')
    sys.exit()
