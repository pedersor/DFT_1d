import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from DFT_1d import ks_dft
from DFT_1d import functionals
from DFT_1d import ext_potentials

# DFT_1D is primarily a DFT solver for one-dimensional (1D) systems.
# First start by defining uniformly spaced set of grid points:

# grid spacing
dx = 0.08
# produce grid of 513 points in the range [-20.48, 20.48]
grids = np.arange(-256, 257) * dx

# define the external potential of the system.
# Here we use a 1D analog of the Li atom (Z=3) using the exponential
# interaction instead of the standard Coulomb interaction.
# Functools.partial is used because our solver expects a callable
# function.
v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=3)

# The potential on a grid can be obtained from the callable:
v_ext_on_grid = v_ext(grids)

# Next, we can set an exchange-correlation (XC) functional to use.
# Here we will use the 1D-LSDA functional developed by:
# [https://doi.org/10.1103/PhysRevB.91.235141]
lsda_xc = functionals.ExponentialLSDFunctional

# initialize the DFT solver for our system and run the SCF calculation.
solver = ks_dft.KS_Solver(
    grids,
    v_ext=v_ext,
    xc=lsda_xc,
    num_electrons=3,
    num_unpaired_electrons=1,
)
solver.solve_self_consistent_density(verbose=1)

# check for SCF convergence and report results.
if solver.is_converged():
  print()
  print('Converged results:')
else:
  print()
  warnings.warn('results are not converged!')

# Non-Interacting (Kohn-Sham) Kinetic Energy
print("T_s =", solver.ks_kinetic_energy)

# External Potential Energy
print("V =", solver.ext_potential_energy)

# Hartree Energy
print("U =", solver.hartree_energy)

# Exchange Energy
print("E_x =", solver.exchange_energy)

# Correlation Energy
print("E_c =", solver.correlation_energy)

# Total Energy
print("E =", solver.total_energy)

# plot self-consistent densities
plt.plot(grids, solver.density)
plt.ylabel('$n(x)$', fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.grid(alpha=0.4)
plt.show()
