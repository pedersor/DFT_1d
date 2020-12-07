import numpy as np
import blue_tools
import matplotlib.pyplot as plt
import sys
# from scipy.interpolate import pade
from scipy.optimize import curve_fit


# two column file to np arrays
def txt_file_to_array(file):
    with open(file) as f:
        lines = f.readlines()
        x = [float(line.split()[0]) for line in lines]
        y = [float(line.split()[1]) for line in lines]

    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def get_Vpp(R):
    return 1. / np.abs(R)


class Quantities():
    # exact results
    def __init__(self, R, Etot, V_ext, Ex_plus_U, Ts, Vee):
        self.R = R
        self.Etot = Etot
        self.Ex = -1 * Ex_plus_U
        self.U = 2 * Ex_plus_U
        self.Ts = Ts
        self.Vee = Vee
        self.V_ext = V_ext
        self.U_xc = Vee - self.U
        self.Uc = self.U_xc - self.Ex
        self.Ec = self.Etot - (self.Ts + self.V_ext + self.U + self.Ex)

    def get_blue_results(self, lambda_grid, Vee_blue):
        self.Vee_blue = Vee_blue
        self.blue_U_xc = Vee_blue - self.U
        self.blue_Uc = self.blue_U_xc[-1] - self.Ex
        self.blue_E_xc = self.get_E_xc_adiab(lambda_grid, self.blue_U_xc)
        self.blue_E_c = self.blue_E_xc - self.Ex
        return

    def get_E_xc_adiab(self, lambda_grid, U_xc):
        lambda_grid = np.asarray(lambda_grid)

        popt, pcov = curve_fit(pade, lambda_grid, U_xc)

        lambda_grid_interp = np.linspace(0, 1, 100)

        '''
        plt.plot(lambda_grid, U_xc)
        plt.plot(lambda_grid_interp, pade(lambda_grid_interp, *popt))
        plt.title('H$_2$, $U_{xc}(\lambda)$ for $R = ' + str(R) + '$')
        plt.ylabel('$U_{xc}(\lambda)$', fontsize=16)
        plt.xlabel('$\lambda$', fontsize=16)
        plt.show()
        '''

        if self.R == 2.6:
            self.Vee_blue = U_xc[-1] + self.U
            return np.trapz(U_xc, lambda_grid)

        E_xc = np.trapz(pade(lambda_grid_interp, *popt), lambda_grid_interp)

        if self.R > 4:
            self.Vee_blue = pade(lambda_grid_interp, *popt)[-1] + self.U
        else:
            self.Vee_blue = U_xc[-1] + self.U

        # E_xc = np.sum(U_xc) / len(U_xc)  # simple averaging
        return E_xc


def pade(x, a, b, c, d):
    return (a + b * x) / (d + c * x)


# exact results
R_grid, T = txt_file_to_array('H2_from_srwhite/T.dat')
R_grid, T_s = txt_file_to_array('H2_from_srwhite/T_s.dat')
R_grid, U_plus_Ex = txt_file_to_array('H2_from_srwhite/U_plus_Ex.dat')
R_grid, E = txt_file_to_array('H2_from_srwhite/E.dat')
R_grid, Vee = txt_file_to_array('H2_from_srwhite/Vee.dat')

V_ext = E - (2 * T + Vee)
E_xc = E - (2 * T_s + V_ext + 2 * U_plus_Ex)
U_c = Vee - U_plus_Ex
T_c = 2 * T - 2 * T_s
E_c = E_xc - (-U_plus_Ex)
U_xc = (-U_plus_Ex) + U_c
E_x = -U_plus_Ex

# blue lam = 1 results
gam = '1or_s'
R, V_ee_blue_lam_1 = txt_file_to_array(
    'H2_from_srwhite/Vee_blue_gam_' + gam + '.dat')

# blue adiab results
lambdas, Vee_blue_adiab = txt_file_to_array(
    'H2_from_srwhite/Vee_blue_adiab.dat')

# append exact results
h2_results = []
for i, R in enumerate(R_grid):
    h2_R = Quantities(R, E[i], V_ext[i], U_plus_Ex[i], 2 * T_s[i], Vee[i])
    h2_results.append(h2_R)

# append blue adiab results
lambdas_in_dat = [0.1, 0.3, 0.5, 0.7]
curr_idx = 0
for i, R in enumerate(R_grid):
    # lam = 0 result
    Vee_blue = [U_plus_Ex[i]]
    # Vee_blue = []

    for lam in lambdas_in_dat:
        Vee_blue.append(Vee_blue_adiab[curr_idx])
        curr_idx += 1

    if R < 5:
        # append lam = 1 result
        Vee_blue.append(V_ee_blue_lam_1[i])

        lambda_grid = [0, 0.1, 0.3, 0.5, 0.7, 1]

    else:
        lambda_grid = [0, 0.1, 0.3, 0.5, 0.7]

    Vee_blue = np.asarray(Vee_blue)

    h2_R = h2_results[i]
    h2_R.get_blue_results(lambda_grid, Vee_blue)
    h2_results[i] = h2_R

# dissociation plots
two_H_atoms_E = -2 * 0.5

E_blue = [h2_R.Ts + h2_R.V_ext + h2_R.U + h2_R.blue_E_xc for h2_R in h2_results]
plt.plot(R_grid, E_blue + get_Vpp(R_grid) - two_H_atoms_E)

E_exact = np.asarray(
    [h2_R.Ts + h2_R.V_ext + h2_R.U + h2_R.Ex + h2_R.Ec for h2_R in h2_results])
plt.plot(R_grid, E_exact + get_Vpp(R_grid) - two_H_atoms_E, color='black')
plt.ylabel('$E_0(R)$', fontsize=16)
plt.xlabel('$R$', fontsize=16)
plt.show()

# write to multi-column file
out_column = [R_grid, E_blue + get_Vpp(R_grid) - two_H_atoms_E,
              E_exact + get_Vpp(R_grid) - two_H_atoms_E]
with open("fig_1_b.dat", "w") as file:
    file.write("R\t E_0_CP \t E_0_exact \n")
    for x in zip(*out_column):
        R_print = x[0]
        to_print = [format(val, '.6f') for val in x[1:]]

        file.write("{0}\t{1}\t{2}\n".format(R_print, *to_print))

# E_c plots

E_c_blue = [h2_R.blue_E_c for h2_R in h2_results]
E_c_exact = [h2_R.Ec for h2_R in h2_results]

plt.plot(R_grid, E_c_blue)
plt.plot(R_grid, E_c_exact, color='black')
plt.ylabel('$E_c(R)$', fontsize=16)
plt.xlabel('$R$', fontsize=16)
plt.show()

# E_x plots

E_x_exact = [h2_R.Ex for h2_R in h2_results]

plt.plot(R_grid, E_x_exact, color='black')
plt.ylabel('$E_x(R)$', fontsize=16)
plt.xlabel('$R$', fontsize=16)
plt.show()

# tables

for h2_R in h2_results:
    blue_tools.table_print(h2_R.R)
    blue_tools.table_print(h2_R.Ex)
    blue_tools.table_print(h2_R.Vee_blue)
    blue_tools.table_print(h2_R.blue_Uc)
    blue_tools.table_print(h2_R.Uc)
    blue_tools.table_print(h2_R.blue_E_c)
    blue_tools.table_print(h2_R.Ec, last_in_row=True)
