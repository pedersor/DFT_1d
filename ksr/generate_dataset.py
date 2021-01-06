import numpy as np
import ks_dft, functionals, ext_potentials
import functools
import os
import sys


class LDA_atom_dataset():
  """Obtain dataset for LDA-calculated systems. Current support is for atoms.
  """

  def __init__(self, grids, selected_ions, locations=None):
    self.grids = grids
    self.selected_ions = selected_ions

    if locations is None:
      self.locations = [[0]] * len(self.selected_ions)
    else:
      self.locations = locations

    # output quantities
    self.num_electrons = []
    self.nuclear_charges = []
    self.total_energies = []
    self.densities = []
    self.external_potentials = []
    self.xc_energies = []
    self.xc_energy_densities = []
    self.xc_potentials = []

  def run_lda_selected_ions(self):
    lda_xc = functionals.ExponentialLDAFunctional(grids=self.grids)
    for ((nuclear_charge, num_electron), center) in zip(self.selected_ions,
                                                        self.locations):
      v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge,
                                center=center)
      solver = ks_dft.Spinless_KS_Solver(self.grids, v_ext=v_ext,
                                         xc=lda_xc,
                                         num_electrons=num_electron)
      solver.solve_self_consistent_density()
      print(
        'finished: (Z, N_e) = (' + str(nuclear_charge) + ',' +
        str(num_electron) + ')')

      self.num_electrons.append(num_electron)
      self.nuclear_charges.append([nuclear_charge])
      self.total_energies.append(solver.E_tot)
      self.densities.append(solver.density)
      self.external_potentials.append(v_ext(self.grids))
      self.xc_energies.append(solver.E_x + solver.E_c)
      self.xc_energy_densities.append(
        solver.xc.xc_energy_density(solver.density))
      self.xc_potentials.append(solver.xc.v_xc(solver.density))

  def run_lsda_selected_ions(self):
    lda_xc = functionals.ExponentialLSDFunctional(grids=self.grids)
    for ((nuclear_charge, num_electron), center) in zip(self.selected_ions,
                                                        self.locations):
      v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge,
                                center=center)
      solver = ks_dft.KS_Solver(self.grids, v_ext=v_ext,
                                xc=lda_xc,
                                num_electrons=num_electron)
      solver.solve_self_consistent_density()
      print(
        'finished: (Z, N_e) = (' + str(nuclear_charge) + ',' +
        str(num_electron) + ')')

      self.num_electrons.append(num_electron)
      self.nuclear_charges.append([nuclear_charge])
      self.total_energies.append(solver.E_tot)
      self.densities.append(solver.density)
      self.external_potentials.append(v_ext(self.grids))
      self.xc_energies.append(solver.E_x + solver.E_c)

      xc_energy_density = solver.xc.e_x(
        solver.density, solver.zeta) + solver.xc.e_c(
        solver.density, solver.zeta)
      self.xc_energy_densities.append(xc_energy_density / solver.density)

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
    np.save(os.path.join(out_dir, 'xc_energy_densities.npy'),
            self.xc_energy_densities)
    np.save(os.path.join(out_dir, 'xc_potentials.npy'),
            self.xc_potentials)


if __name__ == '__main__':
  h = 0.08
  grids = np.arange(-256, 257) * h

  # ions are identified by: atomic number Z, number of electrons
  selected_ions = [(1, 1), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2),
                   (3, 3), (4, 3), (4, 4)]

  out_dir = os.path.join('ions', 'basic_all')

  dataset = LDA_atom_dataset(grids, selected_ions)
  dataset.run_lsda_selected_ions()
  print(dataset.total_energies)
  print(dataset.xc_energies)

  dataset.save_dataset(out_dir=out_dir)
