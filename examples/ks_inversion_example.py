import pathlib

import numpy as np
import matplotlib.pyplot as plt

from DFT_1d import ks_inversion

if __name__ == '__main__':
  example_dir = pathlib.Path('../data/molecules/h_be_h')
  idx = 50

  f_v_Z_fn = lambda _: np.load(example_dir / 'external_potentials.npy')[idx]
  f_tar_density = np.load(example_dir / 'densities.npy')[idx]
  num_electrons = np.load(example_dir / 'num_electrons.npy')[idx]
  f_grids = np.load(example_dir / 'grids.npy')

  ksi = ks_inversion.two_iter_KS_inversion(f_grids, f_v_Z_fn, f_tar_density,
                                           num_electrons)

  f_v_XC = ksi.f_v_XC
  step_list = ksi.step_list
  cost_list = ksi.cost_list
  truncation = ksi.truncation

  # Example plot
  v_s = ksi._get_v_eff()
  n = ksi.f_density
  plt.plot(f_grids, f_tar_density, label='exact $n$')
  plt.plot(f_grids, v_s, label='inversion $v_s$')
  plt.plot(f_grids, n, '--', label='inversion $n$')
  plt.legend()
  plt.show()
