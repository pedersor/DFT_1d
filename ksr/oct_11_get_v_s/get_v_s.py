import pathlib

import numpy as np
import matplotlib.pyplot as plt

from DFT_1d import ks_inversion 

if __name__ == '__main__':
    base_dir = pathlib.Path('../data/molecules/relaxed_all')
    
    external_potentials = np.load(base_dir / 'external_potentials.npy')
    f_tar_density = np.load(base_dir / 'densities.npy')
    num_electrons = np.load(base_dir / 'num_electrons.npy')
    num_unpaired_electrons = np.load(base_dir / 'num_unpaired_electrons.npy')
    latex_symbols = np.load(base_dir / 'latex_symbols.npy')
    f_grids = np.load(base_dir / 'grids.npy')
    
    num_samples = len(num_electrons)
    ncols = 4
    nrows = int(np.ceil(num_samples / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    
    xc_potentials = []
    for i in range(num_samples):
        if num_unpaired_electrons[i] != 0:
            print(f'skipped {latex_symbols[i]}')
            xc_potentials.append(np.zeros_like(f_grids))
            continue

        print(f'running {latex_symbols[i]}')

        ax = axs.reshape(-1)[i]

        ext_pot = lambda _: external_potentials[i]
        ksi = ks_inversion.two_iter_KS_inversion(f_grids, ext_pot, f_tar_density[i], num_electrons[i])

        v_xc = ksi.f_v_XC
        step_list = ksi.step_list
        cost_list = ksi.cost_list
        truncation = ksi.truncation
        v_s = ksi._get_v_eff()
        n = ksi.f_density

        # plots
        ax.plot(f_grids, f_tar_density[i], label='exact $n$')
        ax.plot(f_grids, v_s, label='inversion $v_s$')
        ax.plot(f_grids, v_xc, label='inversion $v_{{xc}}$')
        ax.plot(f_grids, n, '--', label='inversion $n$')
        
        # potentials
        xc_potentials.append(v_xc)

    
    np.save(base_dir / "exact_xc_potentials.npy", np.asarray(xc_potentials))

    fig.tight_layout()
    fig.show()
