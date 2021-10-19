import numpy as np
import sys

A = 1.071295
kpiv = 2.385345


def potpeval(x, ion_positions, nuc_charges):
    Vext = 0
    for i, ion_pos in enumerate(ion_positions):
        Vext += -nuc_charges[i] * A * np.exp(-np.abs(x - ion_pos) / kpiv)

    return Vext


# to output
densities = []  # total density nup + ndown
total_energies = []  # total energies E = T + Vee + Vep
potentials = []  # Vext
locations = []  # locations of ions
nuclear_charges = []  # nuclear charges e.g. [1,1] for H2

# calculation parameters
N = 513  # total number of grids
Nc = 257  # position of center site of the grid
h = 0.08  # grid spacing
grids = np.arange(-(Nc - 1), Nc) * h

Nel = 2  # number of electrons
nuc_charges = [2]
N_locations = len(nuc_charges)


# get density and energy -------------
n = []
for line in reversed(list(open("output.txt"))):
    if 'Etot' in line:
        continue
    elif 'E' in line:
        E = float(line[16:])
        break
    else:
        n.append(float(line[-23:]))

n.reverse()
n = np.asarray(n)
densities.append(n / h)
total_energies.append(E)

# get Vext, ion positions, nuclear_charges -----------------

R = 0
j = 0

nuclear_charges.append(nuc_charges)

ion_positions = h*np.asarray([Nc+j])
locations.append([j*h])
grids_N = h * np.linspace(1, N, N)

potentials.append(potpeval(grids_N, ion_positions, nuc_charges))

densities = np.asarray(densities)
np.save("out_data/densities", densities)

total_energies = np.asarray(total_energies)
print(total_energies)
np.save("out_data/total_energies", total_energies)

potentials = np.asarray(potentials)
print(potentials)
np.save("out_data/potentials", potentials)

locations = np.asarray(locations)
print(locations)
np.save("out_data/locations", locations)

nuclear_charges = np.asarray(nuclear_charges)
print(nuclear_charges)
np.save("out_data/nuclear_charges", nuclear_charges)
