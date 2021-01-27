import sys
import os
import numpy as np
import functools

sys.path.append('../')
from DFT_1d import ks_dft, hf_scf, functionals, ext_potentials


class Generate_dataset():
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
    self.num_unpaired_electrons = []
    self.nuclear_charges = []
    self.total_energies = []
    self.densities = []
    self.external_potentials = []
    self.xc_energies = []
    self.xc_energy_densities = []

  def run_lda_selected_ions(self):
    lda_xc = functionals.ExponentialLDAFunctional
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
      self.num_unpaired_electrons.append(0)
      self.nuclear_charges.append([nuclear_charge])
      self.total_energies.append(solver.total_energy)
      self.densities.append(solver.density)
      self.external_potentials.append(v_ext(self.grids))
      self.xc_energies.append(solver.exchange_energy +
                              solver.correlation_energy)
      self.xc_energy_densities.append(
        solver.xc.xc_energy_density(solver.density))

  def run_lsda_selected_ions(self):
    lda_xc = functionals.ExponentialLSDFunctional
    for ((nuclear_charge, num_electron), center) in zip(self.selected_ions,
                                                        self.locations):

      if num_electron % 2 == 0:
        num_unpaired_electrons = 0
      else:
        num_unpaired_electrons = 1

      v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge,
                                center=center)
      solver = ks_dft.KS_Solver(self.grids, v_ext=v_ext,
                                xc=lda_xc,
                                num_electrons=num_electron,
                                num_unpaired_electrons=num_unpaired_electrons)
      solver.solve_self_consistent_density()
      print(
        'finished: (Z, N_e) = (' + str(nuclear_charge) + ',' +
        str(num_electron) + ')')

      self.num_electrons.append(num_electron)
      self.num_unpaired_electrons.append(num_unpaired_electrons)
      self.nuclear_charges.append([nuclear_charge])
      self.total_energies.append(solver.total_energy)
      self.densities.append(solver.density)
      self.external_potentials.append(v_ext(self.grids))
      self.xc_energies.append(solver.exchange_energy +
                              solver.correlation_energy)

      xc_energy_density = solver.xc.e_x(
        solver.density, solver.zeta) + solver.xc.e_c(
        solver.density, solver.zeta)
      self.xc_energy_densities.append(xc_energy_density / solver.density)

  def run_hf_selected_ions(self):
    exponential_hf = functionals.ExponentialHF
    for ((nuclear_charge, num_electron), center) in zip(self.selected_ions,
                                                        self.locations):

      if num_electron % 2 == 0:
        num_unpaired_electrons = 0
      else:
        num_unpaired_electrons = 1

      v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=nuclear_charge,
                                center=center)
      solver = hf_scf.HF_Solver(self.grids, v_ext=v_ext,
                                hf=exponential_hf,
                                num_electrons=num_electron,
                                num_unpaired_electrons=num_unpaired_electrons)
      solver.solve_self_consistent_density(verbose=1,
                                           energy_converge_tolerance=1e-4)
      print(
        'finished: (Z, N_e) = (' + str(nuclear_charge) + ',' +
        str(num_electron) + ')')

      self.num_electrons.append(num_electron)
      self.num_unpaired_electrons.append(num_unpaired_electrons)
      self.nuclear_charges.append([nuclear_charge])
      self.total_energies.append(solver.total_energy)
      self.densities.append(solver.density)
      self.external_potentials.append(v_ext(self.grids))
      self.xc_energies.append(solver.exchange_energy)

  def save_dataset(self, out_dir):
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    np.save(os.path.join(out_dir, 'grids.npy'), self.grids)
    np.save(os.path.join(out_dir, 'locations.npy'), self.locations)
    np.save(os.path.join(out_dir, 'num_electrons.npy'), self.num_electrons)
    np.save(os.path.join(out_dir, 'num_unpaired_electrons.npy'),
            self.num_unpaired_electrons)
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

    if self.xc_energy_densities:
      np.save(os.path.join(out_dir, 'xc_energy_densities.npy'),
              self.xc_energy_densities)


if __name__ == '__main__':
  """Get dataset for KSR calculations."""
  h = 0.08
  grids = np.arange(-256, 257) * h

  # ions are identified by: atomic number Z, number of electrons
  selected_ions = [(1, 1), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2),
                   (3, 3), (4, 3), (4, 4)]

  dataset = Generate_dataset(grids, selected_ions)
  dataset.run_hf_selected_ions()

  print()
  print('Total energies:')
  print(dataset.total_energies)
  print('XC energies:')
  print(dataset.xc_energies)
  print('num_electrons')
  print(dataset.num_electrons)
  print('num_unpaired_electrons')
  print(dataset.num_unpaired_electrons)
  print('nuclear charges')
  print(dataset.nuclear_charges)

  out_dir = os.path.join('ions', 'hf')
  dataset.save_dataset(out_dir=out_dir)
