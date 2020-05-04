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


def txt_file_to_2d_array(file, grids):
    array_2d = []
    with open(file) as f:
        lines = f.readlines()

        counter = 0
        array_1d = []
        for line in lines:
            array_1d.append(float(line.split()[2]))

            counter += 1
            if counter == len(grids):
                array_1d = np.asarray(array_1d)
                array_2d.append(array_1d)
                counter = 0
                array_1d = []

    array_2d = np.asarray(array_2d)
    return array_2d


def get_P_r_rp_idx(P_r_rp, n, x_idx, h):
    P_r_rp_idx = P_r_rp[x_idx]

    P_r_rp_idx[x_idx] = P_r_rp_idx[x_idx] - n[x_idx] * h

    P_r_rp_idx = P_r_rp_idx / (h * h)
    return P_r_rp_idx


if __name__ == '__main__':
    h = 0.08
    grids = np.arange(-256, 257) * h

    # (x = 0) or specific values
    # exact ----------------
    P_r_rp = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]

    x_idx = 256  # e.g. (x = 0)
    P_r_rp_idx = get_P_r_rp_idx(P_r_rp, n=n_dmrg, x_idx=x_idx, h=h)

    print('n_dmrg[x_idx] ', n_dmrg[x_idx])
    print('integral check: n_dmrg = ', np.sum(n_dmrg) * h)

    print('integral check: P_r_rp_idx = ', np.sum(P_r_rp_idx) * h)

    # blue ----------------------------

    n2_r0 = np.load('n_r0_0.npy')[0][x_idx]

    print('integral check: n2_r0 = ', np.sum(n2_r0) * h)
    print('integral check: (P_r_rp_idx / n_dmrg[x_idx]) = ',
          np.sum((P_r_rp_idx / n_dmrg[x_idx])) * h)

    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]), label='$P^{exact}(0,u)/n(0)$')
    plt.plot(grids, (n2_r0), label='$n^{Blue}_0(u)$')
    plt.xlabel('u', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg,
             label='$n_{xc}(0,u)$')
    plt.plot(grids, (n2_r0) - n_dmrg, label='$n^{Blue}_{xc}(0,u)$')
    plt.xlabel('u', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]) - n_dmrg / 2,
             label='$n_{c}(0,u)$')
    plt.plot(grids, (n2_r0) - n_dmrg / 2, label='$n^{Blue}_{c}(0,u)$')
    plt.xlabel('u', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    sys.exit()

    # compare v_s of CP

    # on-top holes (x = x') --------------------
    P_r_rp = np.load('P_r_rp.npy')
    n_dmrg = np.load('densities.npy')[0]

    P_r_rp_ontop = []
    for x_idx in range(len(grids)):
        P_r_rp_idx = P_r_rp[x_idx]

        P_r_rp_idx[x_idx] = P_r_rp_idx[x_idx] - n_dmrg[x_idx] * h

        P_r_rp_idx = P_r_rp_idx / (h * h)

        P_r_rp_ontop.append(P_r_rp_idx[x_idx])

    P_r_rp_ontop = np.asarray(P_r_rp_ontop)

    plt.plot(grids, P_r_rp_ontop / (n_dmrg * n_dmrg))
    plt.show()
    sys.exit()

    # blue ----------------------------

    n2_r0 = np.load('n_r0_0.npy')[0][x_idx]

    print('integral check: n2_r0 = ', np.sum(n2_r0) * h)
    print('integral check: (P_r_rp_idx / n_dmrg[x_idx]) = ',
          np.sum((P_r_rp_idx / n_dmrg[x_idx])) * h)

    plt.plot(grids, (P_r_rp_idx / n_dmrg[x_idx]), label='$P^{exact}(0,x)/n(0)$')
    plt.plot(grids, (n2_r0), label='$n^{Blue}_0(x)$')
    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.4)
    plt.show()

    sys.exit()
