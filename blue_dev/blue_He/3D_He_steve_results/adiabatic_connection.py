import numpy as np
import matplotlib.pyplot as plt

lam_list = [0, 0.1, 0.3, 0.5, 0.7, 1]
Vee_list = [1.02452, 1.014009395, 0.99285862, 0.9731345, 0.9547283, 0.9294]

U = 2 * 1.02452
Ex = -1.02452

U_xc_list = [Vee - U for Vee in Vee_list]

E_xc = np.trapz(U_xc_list, lam_list)
E_c = E_xc - Ex
print('Ec = ',E_c)
U_c = U_xc_list[-1] - U_xc_list[0]
print('Uc = ', U_c)
T_c = E_c - U_c
print('b =', T_c/np.abs(U_c))

plt.plot(lam_list, U_xc_list)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('$U_{xc}(\lambda)$', fontsize=16)
plt.show()