import matplotlib.pyplot as plt
import numpy as np
import sys


def txt_file_to_array(file, start=0, end=-1):
    # end will not reach end of file... fix..
    # two column file to np arrays
    with open(file) as f:
        lines = f.readlines()[start:end]
        y = [float(line.split()[2]) for line in lines]

    y = np.asarray(y)
    return y


if __name__ == '__main__':
    # exact ----------------
    P_r_rp = txt_file_to_array('exact_pair_density.dat', start=131328,
                               end=131841)

    P_r_rp[256] = P_r_rp[256] - 0.0734800633

    h = 0.08
    grids = np.arange(-256, 257) * h

    P_r_rp = P_r_rp / (h * h)

    print('integral check: P_r_rp = ', np.sum(P_r_rp) * h)

    # blue ----------------------------

    n2_r0 = np.load('n_r0_0.npy')[0][256]
    n_dmrg = np.load('densities.npy')[0]

    print('n_dmrg[256] ', n_dmrg[256])

    print('integral check: n2_r0 = ', np.sum(n2_r0) * h)
    print('integral check: (P_r_rp / n_dmrg[256]) = ', np.sum((P_r_rp / n_dmrg[256])) * h)

    plt.plot(grids, (P_r_rp / n_dmrg[256]), label='$P^{exact}(0,x)/n(0)$')
    plt.plot(grids, (n2_r0), label='$n^{Blue}_0(x)$')

    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()
