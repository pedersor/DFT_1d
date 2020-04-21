import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    n_r0_R = np.load('n_r0_R.npy')
    locations = np.load("../H2_data/locations.npy")
    densities = np.load("../H2_data/densities.npy")

    h = 0.08
    grids = np.arange(-256, 257) * h

    # index corresponding to R = 0.08, 1.52 (equil), 4.00
    R_idx_list = [0, 18, 49]

    # Get the index of 0:
    zero_idx = int(np.where(grids == 0)[0][0])
    # Get the index of 10:
    ten_idx = int(np.where(grids == 10.0)[0][0])

    # first plot ----
    R_idx = R_idx_list[0]

    R_locations = locations[R_idx]
    R_locations.sort()
    R_grid_idx = int(np.where(grids == R_locations[-1])[0][0])

    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][zero_idx],
             label=r'${n}^B_{xc}(0.0, x)$')
    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][R_grid_idx],
             label=r'${n}^B_{xc}(' + str(R_locations[-1]) + ',x)$')
    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][ten_idx],
             label=r'${n}^B_{xc}(10.0, x)$')

    plt.title('1D H$_2$ (R = ' + str(R_locations[1] - R_locations[0]) + ')',
              fontsize=16)
    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

    # second plot ----
    R_idx = R_idx_list[1]

    R_locations = locations[R_idx]
    R_locations.sort()
    R_grid_idx = int(np.where(grids == R_locations[-1])[0][0])

    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][zero_idx],
             label=r'${n}^B_{xc}(0.0, x)$')
    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][R_grid_idx],
             label=r'${n}^B_{xc}(' + str(R_locations[-1]) + ',x)$')
    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][ten_idx],
             label=r'${n}^B_{xc}(10.0, x)$')

    plt.title('1D H$_2$ (R = ' + str(R_locations[1] - R_locations[0]) + ')',
              fontsize=16)
    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

    # third plot ----
    R_idx = R_idx_list[2]

    R_locations = locations[R_idx]
    R_locations.sort()
    R_grid_idx = int(np.where(grids == R_locations[-1])[0][0])

    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][zero_idx],
             label=r'${n}^B_{xc}(0.0, x)$')
    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][R_grid_idx],
             label=r'${n}^B_{xc}(' + str(R_locations[-1]) + ',x)$')
    plt.plot(grids, -densities[R_idx] + n_r0_R[R_idx][ten_idx],
             label=r'${n}^B_{xc}(10.0, x)$')

    plt.title('1D H$_2$ (R = ' + str(R_locations[1] - R_locations[0]) + ')',
              fontsize=16)
    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
