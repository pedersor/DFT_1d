import numpy as np
import functools
import ext_potentials
import sys
import matplotlib.pyplot as plt


def get_Vpp(N, R, Z=1, v_ee=functools.partial(ext_potentials.exp_hydrogenic)):
    Vpp = 0
    for i in range(N):
        for j in range(N)[i:]:
            if i == j:
                continue
            Vpp += Z * (-1) * v_ee((i - j) * R)
    return Vpp


def get_v_H_r(grids, n_r, lam=1, v_ee=functools.partial(
    ext_potentials.exp_hydrogenic)):
    N = len(grids)
    dx = np.abs(grids[1] - grids[0])
    v_H_r = np.zeros(N)
    for i in range(N):
        for j in range(N):
            v_H_r[i] += n_r[i][j] * (-1) * v_ee(grids[i] - grids[j])
    v_H_r *= dx
    return lam * v_H_r


def get_Vee(grids, n, v_H_r):
    Vee = 0.5 * np.trapz(n * v_H_r, grids)
    return Vee


# h-4 chain
N = 4
h = 0.08
grids = np.arange(-256, 257) * h

R_idx_lst = np.array([4, 14, 24, 34, 44, 54, 64, 74])
n = np.load('h4_exact/n_lst.npy')
n = np.asarray([n[idx] for idx in R_idx_lst])

E = np.load('h4_exact/E_lst.npy')
E = np.asarray([E[idx] for idx in R_idx_lst])

T_plus_Vext = np.load('h4_exact/T_plus_Vext_lst.npy')
T_plus_Vext = np.asarray([T_plus_Vext[idx] for idx in R_idx_lst])

Vee_exact = np.load('h4_exact/Vee_lst.npy')
Vee_exact = np.asarray([Vee_exact[idx] for idx in R_idx_lst])

V_pp = np.array([get_Vpp(N, (R + 1) * h) for R in R_idx_lst])

plt.plot(R_idx_lst * h, T_plus_Vext + Vee_exact + V_pp, color='black',
         label='exact')

# blue quantities ------------------------------

n_R_r = np.load('n_R_r_out.npy')
Vee_blue_R = []
for i, n_r in enumerate(n_R_r):
    v_H_r = get_v_H_r(grids, n_r)
    Vee_blue = get_Vee(grids, n[i], v_H_r)
    Vee_blue_R.append(Vee_blue)

Vee_blue_R = np.asarray(Vee_blue_R)

plt.plot(R_idx_lst * h, T_plus_Vext + Vee_blue_R + V_pp, color='blue',
         label='pure blue')

plt.ylabel('$E_0(R)$ (a.u.)', fontsize=18)
plt.xlabel('$R$ (a.u.)', fontsize=18)
plt.grid(alpha=0.4)
plt.legend(fontsize=16)
plt.show()
