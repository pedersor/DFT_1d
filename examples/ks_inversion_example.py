import pathlib

import jax
from jax import tree_util
import numpy as np
import matplotlib.pyplot as plt

from DFT_1d import ks_inversion
from DFT_1d import functionals

if __name__ == '__main__':
  example_dir = pathlib.Path('../data/molecules/li_h')
  separation = 5.92  # in units Bohr

  example = {
      'distances': np.load(example_dir / 'distances.npy'),
      'external_potentials': np.load(example_dir / 'external_potentials.npy'),
      'densities': np.load(example_dir / 'densities.npy'),
      'num_electrons': np.load(example_dir / 'num_electrons.npy'),
      'latex_symbols': np.load(example_dir / 'latex_symbols.npy'),
  }
  grids = np.load(example_dir / 'grids.npy')

  example = tree_util.tree_map(
      lambda x: x[example['distances'] == separation][0], example)

  lda_xc = functionals.ExponentialLDAFunctional(grids)
  lda_v_xc = lda_xc.get_xc_potential(example['densities'])

  ksi = ks_inversion.two_iter_KS_inversion(
      grids,
      lambda _: example['external_potentials'],
      example['densities'],
      example['num_electrons'],
      init_v_xc=lda_v_xc,
      mixing_param=0.4,
      max_iters=200,
  )

  v_xc = ksi.f_v_XC
  n = ksi.f_density

  fig, ax = plt.subplots()
  ax.plot(grids, example['densities'], label='exact $n$')
  ax.plot(grids, v_xc, label=r'inversion $v_{XC}$')
  ax.plot(grids, n, '--', label='inversion $n$')
  ax.plot(grids, lda_v_xc, label=r'$v^{LDA}_{XC}$')

  ax.set_xlabel(r'$x$')
  ax.set_xlim(-10, 10)
  plt.legend()
  plt.show()
