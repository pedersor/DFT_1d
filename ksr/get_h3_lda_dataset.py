import os
import numpy as np
import sys

sys.path.append('../')

from jax import tree_util
from jax_dft import xc
from jax_dft import spin_scf
from jax_dft import utils

import matplotlib.pyplot as plt
import time

h = 0.08
grids = np.arange(-256, 257) * h

locations = np.load('h3_locations.npy')
num_samples = len(locations)

distances = utils.compute_distances_between_nuclei(locations, (0, 1))
nuclear_charges = np.array([[1, 1, 1]] * num_samples)
num_electrons = [3] * num_samples
num_unpaired_electrons = [1] * num_samples

# setup LSDA functional
xc_energy_density_fn = tree_util.Partial(
  xc.get_lsda_xc_energy_density_fn(), params=None)

total_energies = []
densities = []
time_per_molecule = []
for i in range(num_samples):
  start_time = time.time()

  lsda_ksdft = spin_scf.kohn_sham(
    locations=locations[i],
    nuclear_charges=nuclear_charges[i],
    num_electrons=num_electrons[i],
    num_unpaired_electrons=num_unpaired_electrons[i],
    num_iterations=30,
    grids=grids,
    xc_energy_density_fn=xc_energy_density_fn,
    interaction_fn=utils.exponential_coulomb,
    # The initial density of KS self-consistent calculations.
    initial_density=None,
    alpha=0.5,
    alpha_decay=0.9,
    # reflection symmetry not supported yet in `spin_scf`
    enforce_reflection_symmetry=False,
    num_mixing_iterations=1,
    density_mse_converge_tolerance=-1)

  time_elapsed = time.time() - start_time
  time_per_molecule.append(time_elapsed)
  if i == 0:
    est_time_remain = time_elapsed * num_samples
    hours, rem = divmod(est_time_remain, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Estimated time to complete: ")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

  total_energies.append(lsda_ksdft.total_energy[-1])
  densities.append(lsda_ksdft.density[-1])

total_energies = np.asarray(total_energies)
densities = np.asarray(densities)
# dissociation plot results
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
  locations,
  nuclear_charges,
  interaction_fn=utils.exponential_coulomb)
plt.plot(distances, total_energies + nuclear_energy)
plt.xlabel(r'$R\,\,\mathrm{(Bohr)}$')
plt.ylabel(r'$E+E_\mathrm{nn}\,\,\mathsf{(Hartree)}$')
plt.savefig("lsda_h3_dissociation.pdf", bbox_inches='tight')

out_dir = 'molecule_dissociation/h3/lsda/'
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

np.save(os.path.join(out_dir, 'time_per_molecule.npy'), time_per_molecule)
np.save(os.path.join(out_dir, 'locations.npy'), locations)
np.save(os.path.join(out_dir, 'total_energies.npy'), total_energies)
np.save(os.path.join(out_dir, 'densities.npy'), densities)
np.save(os.path.join(out_dir, 'nuclear_charges.npy'), nuclear_charges)
np.save(os.path.join(out_dir, 'num_electrons.npy'), num_electrons)
np.save(os.path.join(out_dir, 'num_unpaired_electrons.npy'),
        num_unpaired_electrons)
