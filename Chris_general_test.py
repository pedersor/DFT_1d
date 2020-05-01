import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import six
from six.moves import range
import copy
import timeit
import functools

import single_electron
import ext_potentials

'''
grids = np.linspace(-4, 4, 10)
solver = single_electron.EigenSolver(grids, potential_fn=functools.partial(ext_potentials.poschl_teller, lam=1), boundary_condition='closed', n_point_stencil=5)
solver._set_matrices()
print(np.round(solver._t_mat*(-2)*solver.dx**2, 2))
'''
