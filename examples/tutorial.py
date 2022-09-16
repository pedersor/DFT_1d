import functools
import warnings
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from DFT_1d import ks_dft
from DFT_1d import functionals
from DFT_1d import ext_potentials
from DFT_1d import ks_inversion

# DFT_1D is primarily a DFT solver for one-dimensional (1D) systems.
# First start by defining uniformly spaced set of grid points:

# grid spacing
dx = 0.08
# produce grid of 513 points in the range [-20.48, 20.48]
grids = np.arange(-256, 257) * dx

# define the external potential of the system.
# Here we use a 1D analog of the Be atom (Z=4) using the exponential
# interaction instead of the standard Coulomb interaction.
# Functools.partial is used because our solver expects a callable
# function.
v_ext = functools.partial(ext_potentials.exp_hydrogenic, Z=4)

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
    num_electrons=4,
    num_unpaired_electrons=0,
)
solver.solve_self_consistent_density(verbose=1)

# check for SCF convergence and report results.
if solver.is_converged():
  print()
  print('Converged LSDA results:')
else:
  print()
  warnings.warn('results are not converged!')

# Non-Interacting (Kohn-Sham) Kinetic Energy
print("T_s(LSDA) =", solver.ks_kinetic_energy)

# External Potential Energy
print("V(LSDA) =", solver.ext_potential_energy)

# Hartree Energy
print("U(LSDA) =", solver.hartree_energy)

# XC Energy
print("E_xc(LSDA) =", solver.exchange_energy + solver.correlation_energy)

# Total Energy
print("E(LSDA) =", solver.total_energy)

# plot self-consistent density and Kohn-Sham (KS) potential.
fig, ax = plt.subplots()
ax.plot(grids, solver.density, label='$n^{LSDA}(x)$')
ax.plot(grids, solver.v_s_up(grids), label='$v^{LSDA}_s(x)$')
ax.set_xlabel('$x$', fontsize=16)
ax.grid(alpha=0.4)
ax.legend(fontsize=16)
ax.set_xlim(-10, 10)
fig.savefig('lsda_results.pdf', bbox_inches='tight')

# We can also compare to exact results (obtained from DMRG).
# First load in the dataset:
dmrg_data = pathlib.Path('../data/ions/dmrg')
dmrg_data = {
    'densities': np.load(dmrg_data / 'densities.npy'),
    'latex_symbols': np.load(dmrg_data / 'latex_symbols.npy'),
    'total_energies': np.load(dmrg_data / 'total_energies.npy')
}
# Find data for the Be system:
dmrg_data = {
    key: val[dmrg_data['latex_symbols'] == 'Be'][0]
    for key, val in dmrg_data.items()
}
exact_density = dmrg_data['densities']
exact_tot_energy = dmrg_data['total_energies']

# Run a KS inversion to obtain an exact KS potential
ksi = ks_inversion.two_iter_KS_inversion(
    grids,
    lambda _: v_ext_on_grid,
    exact_density,
    num_electrons=4,
    mixing_param=0.4,
)
v_s_exact = ksi.get_v_s()

ax.plot(grids, exact_density, '--', color='blue', label='$n^{exact}(x)$')
ax.plot(grids, v_s_exact, '--', color='red', label='$v^{exact}_s(x)$')
ax.legend(fontsize=16)
fig.savefig('exact_results.pdf', bbox_inches='tight')

# The inversion is also useful to extract exact energy quantities:
t_s_exact = ksi.KE
u_exact = lsda_xc(grids).get_hartree_energy(exact_density)
# compute integrals simply using rectangular integration
v_exact = np.sum(v_ext_on_grid * exact_density) * dx
xc_exact = exact_tot_energy - t_s_exact - u_exact - v_exact

print()
print('Exact results:')
print("T_s(exact) =", t_s_exact)
print("V(exact) =", v_exact)
print("U(exact) =", u_exact)
print("E_xc(exact) =", xc_exact)
print("E(exact) =", exact_tot_energy)
