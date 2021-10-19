import numpy as np


def exp_hydrogenic(grids, A=1.071295, k=(1. / 2.385345), Z=1):
    vp = -Z * A * np.exp(-k * np.abs(grids))
    return vp


def linear_ext_potential(grids):
    return 20*np.abs(grids)

def harmonic_osc(grids, k=0.25):
	return 0.5*k*(grids**2)

def get_nuc_positions(grids, num_electrons, R):
    zero_idx = np.where(grids == 0)[0][0]

    nuc_positions = []  # positions of nuclei
    if R % 2 == 0:
        nuc_L = zero_idx - R / 2  # position of the first nucleus
        nuc_R = zero_idx + R / 2  # position of the second nucleus
        nuc_positions.append(nuc_L)
        nuc_positions.append(nuc_R)

        for i in range(int(num_electrons / 2) - 1):
            nuc_positions.append(nuc_R + (i + 1) * R)
            nuc_positions.append(nuc_L - (i + 1) * R)
    else:
        nuc_L = zero_idx - (R - 1) / 2
        nuc_R = zero_idx + (R + 1) / 2
        nuc_positions.append(nuc_L)
        nuc_positions.append(nuc_R)

        for i in range(int(num_electrons / 2) - 1):
            nuc_positions.append(nuc_R + (i + 1) * R)
            nuc_positions.append(nuc_L - (i + 1) * R)

    nuc_positions.sort()
    nuc_positions = [int(nuc_position) for nuc_position in nuc_positions]
    return nuc_positions


def get_ham1c(grids, pot):
    N = len(grids)

    finite_diff_coeffs = np.array([30, -16, 1]) / 24 / (h ** 2)

    i_lst = []
    j_lst = []
    ham_lst = []
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i == j:
                i_lst.append(i)
                j_lst.append(j)
                ham_lst.append(finite_diff_coeffs[0] + pot[i - 1])
            elif i - j == 1 or j - i == 1:
                i_lst.append(i)
                j_lst.append(j)
                ham_lst.append(finite_diff_coeffs[1])
            elif i - j == 2 or j - i == 2:
                i_lst.append(i)
                j_lst.append(j)
                ham_lst.append(finite_diff_coeffs[2])

    to_out = np.array([i_lst, j_lst, ham_lst])
    to_out = to_out.T
    with open('Ham1c', 'w') as datafile_id:
        # here you open the ascii file
        np.savetxt(datafile_id, to_out, fmt=['%i', '%i', '%.20f'],
                   header=str(N) + '\n' + str(0), comments='')


def get_Vuncomp(grids, lam=1):
    N = len(grids)

    i_lst = []
    j_lst = []
    vuncomp_lst = []
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            vuncomp = -lam * exp_hydrogenic(grids[i - 1] - grids[j - 1])
            i_lst.append(i)
            j_lst.append(j)
            vuncomp_lst.append(vuncomp)

    to_out = np.array([i_lst, j_lst, vuncomp_lst])
    to_out = to_out.T
    with open('Vuncomp', 'w') as datafile_id:
        # here you open the ascii file
        np.savetxt(datafile_id, to_out, fmt=['%i', '%i', '%.20f'],
                   header=str(N), comments='')


lam = 700
h = 0.08  # grid spacing
grids = np.arange(-256, 257) * h

# set external potential
ext_pot = harmonic_osc(grids)

get_ham1c(grids, ext_pot)
get_Vuncomp(grids, lam)
