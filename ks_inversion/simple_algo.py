import single_electron, ext_potentials, functionals, two_el_exact
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
import functools
import sys


def derivative_mat(N, h):
    mat = np.eye(N)
    idx = np.arange(N)

    A = [0, 2 / 3, -1 / 12]

    for j, A_n in enumerate(A):
        mat[idx[j:], idx[j:] - j] = -A_n
        mat[idx[:-j], idx[:-j] + j] = A_n

    return mat / h


def v_vw(n, D_mat, T):
    return (1 / 8) * (np.dot(D_mat, np.transpose(n)) ** 2) / (n ** 2) - (
            1 / 4) * (-2) * (
                   np.dot(T, np.transpose(n)) / n)


def t_vw(n, D_mat):
    return (1 / 8) * (np.dot(D_mat, np.transpose(n)) ** 2) / (n)


def grided_potential(grids, pot):
    return pot


if __name__ == '__main__':
    dir = 'H4_data'
    densities = np.load(dir + "/densities.npy")
    potentials_ext = np.load(dir + "/potentials.npy")
    locations = np.load(dir + "/locations.npy")

    h = 0.08
    grids = np.arange(-256, 257) * h

    H4_v_s_exact_all = np.load('H4_v_s_exact_all.npy')

    grids_idx = [14, 24, 34, 44, 54, 64, 74]
    H4_v_s_exact_select = []
    for idx in grids_idx:
        #print(locations[idx])
        print('R = ', locations[idx][1] - locations[idx][0])
        H4_v_s_exact_select.append(H4_v_s_exact_all[idx])



    H4_v_s_exact_select = np.asarray(H4_v_s_exact_select)
    np.save('H4_v_s_exact_select.npy', H4_v_s_exact_select)
    print(H4_v_s_exact_select.shape)
    
    sys.exit()


    # get H4_v_s_exact_all --------------------------------------

    H4_v_s_exact_all = []
    for R_idx in range(0, 75):

        print('locations : ', locations[R_idx])
        print('R = ', locations[R_idx][1] - locations[R_idx][0])

        n = densities[R_idx]
        v_ext = potentials_ext[R_idx]
        v_H = functionals.hartree_potential(grids, n)
        start, end = two_el_exact.get_truncated_system(n, tol=10 ** (-4))

        grids_trunc = grids[start:end]
        n_trunc = n[start:end]
        v_H_trunc = v_H[start:end]
        v_ext_trunc = v_ext[start:end]

        # get T_matrix
        solver = single_electron.EigenSolver(grids_trunc,
                                             potential_fn=functools.partial(
                                                 grided_potential,
                                                 pot=v_ext_trunc),
                                             boundary_condition='open',
                                             num_electrons=2)
        T = solver.get_kinetic_matrix()
        solver.solve_ground_state()

        E_prev = 2 * solver.total_energy

        D_mat = derivative_mat(len(n_trunc), h)
        v_vw_n_target = v_vw(n_trunc, D_mat, T)
        t_vw_n_target = t_vw(n_trunc, D_mat)

        v_xc_n_plus1 = 0

        for i in range(0, 50):
            v_s_n_plus1 = v_ext_trunc + v_H_trunc + v_xc_n_plus1

            solver = single_electron.EigenSolver(grids_trunc,
                                                 potential_fn=functools.partial(
                                                     grided_potential,
                                                     pot=v_s_n_plus1),
                                                 boundary_condition='open',
                                                 num_electrons=2)
            solver.solve_ground_state()

            E_curr = 2 * solver.total_energy
            E_diff = np.abs(E_prev - E_curr)
            # print('E_diff = ', E_diff)
            E_prev = E_curr

            n_curr = 2 * solver.density
            T_s_curr = 2 * solver.kinetic_energy

            v_vw_n_LDA = v_vw(n_curr, D_mat, T)

            v_xc_n_plus1 = v_xc_n_plus1 + (-v_vw_n_target + v_vw_n_LDA)

            d_ntar_n = (1 / 4) * ((np.sum((n_trunc - n_curr) ** 2) * h) ** (.5))
            # print('$d(n^{target},n^i) $', d_ntar_n)

        v_s_algo = v_ext_trunc + v_H_trunc + v_xc_n_plus1

        v_s_full = two_el_exact.v_s_derivs(v_s_algo[2:-2], n, grids, h,
                                           tol=10 ** (-4))
        # plt.plot(grids, v_s_full)

        # get KS orbitals
        solver = single_electron.EigenSolver(grids,
                                             potential_fn=functools.partial(
                                                 ext_potentials.get_gridded_potential,
                                                 potential=v_s_full),
                                             boundary_condition='open',
                                             num_electrons=4)
        solver.solve_ground_state()

        phi_0 = solver.wave_function[0]
        phi_1 = solver.wave_function[1]
        phi_2 = solver.wave_function[2]
        phi_3 = solver.wave_function[3]

        '''
        # see KS eigenvalues
        e_0 = solver.eigenvalues[0]
        e_1 = solver.eigenvalues[1]
        e_2 = solver.eigenvalues[2]
        e_3 = solver.eigenvalues[3]
    
        print('e_0 = ', format(e_0, '.5f'))
        print('e_1 = ', format(e_1, '.5f'))
        print('e_2 = ', format(e_2, '.5f'))
        print('e_3 = ', format(e_3, '.5f'))
        '''

        '''
        # see orbitals
        plt.plot(grids, phi_0, label='$\phi_0$')
        plt.plot(grids, phi_1, label='$\phi_1$')
        plt.plot(grids, phi_2, label='$\phi_2$')
        plt.plot(grids, phi_3, label='$\phi_3$')
        '''

        if R_idx in [14, 24, 34, 44, 54, 64, 74]:
            # plot densities and v_s_exact
            n_algo = 2 * (phi_0 * phi_0 + phi_1 * phi_1)

            plt.plot(grids, v_s_full, label='v_s_algo')
            plt.plot(grids, n, label='n')
            plt.plot(grids, n_algo, label='n_algo')

            plt.legend()
            plt.show()

        H4_v_s_exact_all.append(v_s_full)

    H4_v_s_exact_all = np.asarray(H4_v_s_exact_all)
    np.save('H4_v_s_exact_all.npy', H4_v_s_exact_all)

    sys.exit()

    # exchange holes, etc. ---------------------------------

    r_idx_dict = {0: 256, 1.04: 256 + 13, 2: 256 + 25}
    r_idx = r_idx_dict[1.04]

    KS_orbitals = [phi_0, phi_1]


    def gamma_s(r_idx):
        gamma_s_array = 0
        for phi in KS_orbitals:
            gamma_s_array += phi[r_idx] * phi

        return gamma_s_array


    n_up = n / 2

    n_exch_hole = - (gamma_s(r_idx) ** 2) / n_up[r_idx]
    n_cp_exch = n_up + n_exch_hole

    exch_long_range_limt = -1 * ((phi_1[r_idx]) ** 2) / n_up[r_idx]
    print(exch_long_range_limt)

    plt.plot(grids, n_exch_hole / n_up,
             label=r'$\tilde n_{exch, \uparrow, x = 1.04}(x \prime )/n_{\uparrow}(x \prime)$')
    plt.plot(grids, exch_long_range_limt * np.ones(len(grids)), '--',
             color='red', alpha=0.4,
             label=r'$-|\phi_{\uparrow}^{HOMO}(1.04)|^2 / n_{\uparrow}(1.04)$')

    plt.xlim(-7.5, 7.5)
    plt.ylim(-1.5, 0.5)
    plt.xlabel(r'$x \prime$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    sys.exit()

    # plot exch holes --------------------------------------------------
    plt.plot(grids, n_cp_exch,
             label=r'$\tilde{n}_{\uparrow, ' + str(
                 grids[r_idx]) + r'}^{\lambda = 0}(\uparrow, x)$')

    solver = single_electron.EigenSolver(grids,
                                         potential_fn=functools.partial(
                                             grided_potential,
                                             pot=v_ext),
                                         boundary_condition='open',
                                         num_electrons=2)
    second_deriv_mat = solver.get_kinetic_matrix()
    plt.plot(grids, np.matmul(.25 * -2 * second_deriv_mat, n_cp_exch),
             label=r'$\frac{d^2}{4dx^2} \tilde{n}_{\uparrow, ' + str(
                 grids[r_idx]) + r'}^{\lambda = 0}(\uparrow, x)$')

    plt.plot(grids, n_exch_hole,
             label=r'$\tilde{n}_{\uparrow, ' + str(
                 grids[
                     r_idx]) + r'}^{\lambda = 0}(\uparrow, x) - n_{\uparrow}(x)$')

    plt.xlim(-20, 20)
    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
