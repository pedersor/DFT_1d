import ext_potentials


def pure_blue_arb_pot_1d(grids, pot, r0, lam=1):
    # pot is the pre-generated external potential of the separated H2
    return pot - lam * ext_potentials.exp_hydrogenic(grids - r0)
