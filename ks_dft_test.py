import ks_dft, dft_potentials, ext_potentials
import numpy as np
import functools

from absl import app
from absl import flags

flags.DEFINE_integer('z', 3, 'Nuclear Charge')
flags.DEFINE_integer('num_electrons', 3, 'Number of electrons in the system.')

FLAGS = flags.FLAGS

def main(argv):
	if len(argv) > 1:
		raise app.UsageError('Too many command-line arguments.')

	A = 1.071295
	k = 1. / 2.385345

	grids = np.linspace(-10, 10, 201)

	v_ext = functools.partial(ext_potentials.exp_hydrogenic, A=A, k=k, a=0, Z=FLAGS.z)
	v_h = functools.partial(dft_potentials.hartree_potential_exp, A=A, k=k, a=0)
	ex_corr = dft_potentials.exchange_correlation_functional(grids=grids, A=A, k=k)

	solver = ks_dft.KS_Solver(grids, v_ext=v_ext, v_h=v_h, xc=ex_corr, num_electrons=FLAGS.num_electrons)
	solver.solve_self_consistent_density()

	# Non-Interacting Kinetic Energy
	print("T_s =", solver.T_s)

	# External Potential Energy
	print("V =", solver.V)

	# Hartree Energy
	print("U =", solver.U)


	# Exchange Energy
	print("E_x =", solver.E_x)

	# Correlation Energy
	print("E_c =", solver.E_c)

	# Total Energy
	print("E =", solver.E_tot)


if __name__ == '__main__':
	app.run(main)
