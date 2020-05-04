import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == '__main__':
    n_r0_R = np.load('../H2_data/n_r0_R.npy')
    n_r0_0 = np.load('n_r0_0.npy')

    print(n_r0_0[0])

    n_r0_R = np.insert(n_r0_R, 0, n_r0_0[0], axis=0)

    print(np.shape(n_r0_R))

    np.save('n_r0_R.npy', n_r0_R)

    sys.exit(0)
    h = 0.08
    grids = np.arange(-256, 257) * h

    # H2 (R >= 0.08)
    densities = np.load("../H2_data/densities.npy")
    potentials = np.load("../H2_data/potentials.npy")
    locations = np.load("../H2_data/locations.npy")
    nuclear_charges = np.load("../H2_data/nuclear_charges.npy")
    total_energies = np.load("../H2_data/total_energies.npy")
    Vee_energies = np.load("../H2_data/Vee_energies.npy")

    '''
    # He (R = 0)
    n = np.load("../He_data/densities.npy")[0]

    v = np.load("../He_data/potentials.npy")[0]
    loc = np.insert(np.load("../He_data/locations.npy")[0], 0, 0.)
    nuc_ch = [1, 1]
    tot_e = np.load('../He_data/total_energies.npy')[0]
    vee = 0.68921094203751598339

    densities = np.insert(densities, 0, n, axis=0)
    potentials = np.insert(potentials, 0, v, axis=0)
    locations = np.insert(locations, 0, loc, axis=0)
    nuclear_charges = np.insert(nuclear_charges, 0, nuc_ch, axis=0)
    total_energies = np.insert(total_energies, 0, tot_e)
    Vee_energies = np.insert(Vee_energies, 0, vee)
    
    '''

    print(np.shape(densities))
    print(np.shape(potentials))
    print(np.shape(locations))
    print(np.shape(nuclear_charges))
    print(np.shape(total_energies))
    print(np.shape(Vee_energies))

    sys.exit()

    np.save("../H2_data/densities.npy", densities)
    np.save("../H2_data/potentials.npy", potentials)
    np.save("../H2_data/locations.npy", locations)
    np.save("../H2_data/nuclear_charges.npy", nuclear_charges)
    np.save("../H2_data/total_energies.npy", total_energies)
    np.save("../H2_data/Vee_energies.npy", Vee_energies)

