import numpy as np


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
