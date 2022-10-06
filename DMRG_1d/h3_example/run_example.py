import os
from shutil import copyfile
import numpy as np
from DFT_1d import utils
import hoppingvmaker


def mkdir_p(dir):
  """Make a directory (dir) if it doesn't exist."""
  if not os.path.exists(dir):
    os.mkdir(dir)


def edit_input_file(separation):
  with open('input', 'r') as file:
    input = file.readlines()

  for i, line in enumerate(input):
    if 'separation' in line:
      input[i] = f'separation = {separation}\n'

  with open('input', 'w') as file:
    file.writelines(input)


if __name__ == '__main__':
  """ H_3 chain example.
  Note(pedersor): you must change the inital wavefunction in electronBO.cc for 
  other examples.
  """

  h = 0.08  # grid spacing
  grids = np.arange(-256, 257) * h
  # range of separations in Bohr: (min, max)
  separations = np.arange(0, 6, h)
  nuclear_charges = np.array([1, 1, 1])
  num_electrons = 3
  num_unpaired_electrons = 1

  cwd = os.getcwd()

  locations_lst = []
  external_potentials_lst = []
  for sep in separations:
    sep_steps = int(round(float(sep / h)))

    curr_dir = f'R{sep_steps}'
    print(curr_dir)
    mkdir_p(curr_dir)

    # edit the input file
    edit_input_file(sep_steps)

    locations = utils.get_unif_separated_nuclei_positions(
        grids, num_locations=len(nuclear_charges), separation=sep)
    locations_lst.append(locations)

    external_potential = utils.get_atomic_chain_potential(
        grids=grids,
        locations=locations,
        nuclear_charges=nuclear_charges,
    )
    external_potentials_lst.append(external_potential)

    hoppingvmaker.get_ham1c(grids, external_potential)
    hoppingvmaker.get_vuncomp(grids)

    # compress vuncomp to MPO
    os.system('''julia compressMPO.jl Vuncomp''')
    # remove the large uncompressed file
    os.remove('Vuncomp')

    copyfile('input', os.path.join(curr_dir, 'input'))
    copyfile('electronBO.cc', os.path.join(curr_dir, 'electronBO.cc'))
    copyfile('Makefile', os.path.join(curr_dir, 'Makefile'))

    # move files
    os.rename("Ham1c", os.path.join(curr_dir, "Ham1c"))
    os.rename("Vcompressed", os.path.join(curr_dir, "Vcompressed"))

    os.chdir(curr_dir)

    # compile and run
    os.system('make')
    os.system('./electronBO input > output.out')

    # remove large not needed data
    os.remove('electronBO')
    os.remove('electronBO.o')
    os.remove('rm sites')
    os.chdir(cwd)

  # create dataset
  mkdir_p('dataset')

  locations = np.asarray(locations_lst)
  external_potentials = np.asarray(external_potentials_lst)
  num_samples = len(separations)
  num_electrons = (num_electrons * np.ones(num_samples)).astype(int)
  num_unpaired_electrons = (num_unpaired_electrons *
                            np.ones(num_samples)).astype(int)
  nuclear_charges = np.tile(nuclear_charges, reps=(num_samples, 1))
  distances = separations
  distances_x100 = (100 * separations).astype(int)

  np.save('dataset/grids.npy', grids)
  np.save('dataset/locations.npy', locations)
  np.save('dataset/distances.npy', distances)
  np.save('dataset/distances_x100.npy', distances_x100)
  np.save('dataset/external_potentials.npy', external_potentials)
  np.save('dataset/num_electrons.npy', num_electrons)
  np.save('dataset/num_unpaired_electrons.npy', num_unpaired_electrons)
  np.save('dataset/nuclear_charges.npy', nuclear_charges)
