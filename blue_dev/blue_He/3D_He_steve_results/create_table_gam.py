import numpy as np
import blue_tools
import sys


class Quantities():
    # exact results
    def __init__(self, Z, Etot, Ex_plus_U, Ts, Vee):
        self.Z = Z
        self.Etot = Etot
        self.Ex = -1 * Ex_plus_U
        self.U = 2 * Ex_plus_U
        self.Ts = Ts
        self.Vee = Vee
        self.V_ext = (2 * Etot) - Vee
        self.Uxc = Vee - self.U
        self.Uc = self.Uxc - self.Ex
        self.Tc = self.get_T_c_exact(self.Uc)
        self.Ec = self.Uc + self.Tc

    def init_HF_results(self, Etot_HF, Ex_plus_U_HF, Ts_HF):
        self.Etot_HF = Etot_HF
        self.Ex_HF = -1 * Ex_plus_U_HF
        self.U_HF = 2 * Ex_plus_U_HF
        self.Ts_HF = Ts_HF #-1 * Etot_HF  # enforce virial theorem (!)
        self.V_ext_HF = self.Etot_HF - self.Ts_HF - self.U_HF - self.Ex_HF

    def get_blue_results_w_HF(self, Vee_blue_HF):
        self.Vee_blue_HF = Vee_blue_HF
        self.blue_Uxc_HF = Vee_blue_HF - self.U_HF
        self.blue_Uc_HF = self.blue_Uxc_HF - self.Ex_HF
        self.blue_Tc_HF = self.get_T_c_HF(self.blue_Uc_HF)
        self.blue_Ec_HF = self.blue_Uc_HF + self.blue_Tc_HF
        return

    def get_blue_results_exact(self, Vee_blue_exact):
        self.Vee_blue = Vee_blue_exact
        self.blue_Uxc = Vee_blue_exact - self.U
        self.blue_Uc = self.blue_Uxc - self.Ex
        self.blue_Tc = self.get_T_c_exact(self.blue_Uc)
        self.blue_Ec = self.blue_Uc + self.blue_Tc
        return

    def get_T_c_exact(self, Uc):
        T_c = 0.5 * (-2 * self.Ts - self.V_ext - self.U - self.Ex - Uc)
        return T_c

    def get_T_c_HF(self, Uc):
        T_c_HF = 0.5 * (
                -2 * self.Ts_HF - self.V_ext_HF - self.U_HF - self.Ex_HF - Uc)
        return T_c_HF


# exact quantities (using s=0.7, c=0.2 basis)
start = 1
with open('ion_results_all.dat') as f:
    lines = f.readlines()
    Z = [float(line.split()[0]) for line in lines[start:]]
    Etot = [float(line.split()[1]) for line in lines[start:]]
    Ex_plus_U = [float(line.split()[2]) for line in lines[start:]]
    Ts = [2 * float(line.split()[3]) for line in lines[start:]]
    Vee = [float(line.split()[4]) for line in lines[start:]]
    Vee_blue = [float(line.split()[5]) for line in lines[start:]]

with open('ion_results_all_HF.dat') as f:
    lines = f.readlines()
    Etot_HF = [float(line.split()[1]) for line in lines[start:]]
    Ex_plus_U_HF = [float(line.split()[2]) for line in lines[start:]]
    Ts_HF = [float(line.split()[3]) for line in lines[start:]]
    Vee_blue_HF = [float(line.split()[4]) for line in lines[start:]]

Z_results = []
# cyrus is exact
cyrus_Ts = [0.499869, 2.867082, 7.240085, 13.614084, 32.363072]
for i, ion in enumerate(Z):
    Z_Quantities = Quantities(ion, Etot[i], Ex_plus_U[i], cyrus_Ts[i], Vee[i])
    Z_Quantities.get_blue_results_exact(Vee_blue[i])

    Z_Quantities.init_HF_results(Etot_HF[i], Ex_plus_U_HF[i], Ts_HF[i])
    Z_Quantities.get_blue_results_w_HF(Vee_blue_HF[i])

    Z_results.append(Z_Quantities)

# HF results
print(
    ' Z   Ex(HF)      Vee_blue_HF      Uc_blue_HF       Uc(exact)   Uc(error)   Ec(blue_HF)    Ec(exact)  Ec(error) ')
for i, ion in enumerate(Z_results):
    Z = str(ion.Z)
    blue_tools.table_print(Z)

    blue_tools.table_print(ion.Ex_HF, round_to_dec=4)

    blue_tools.table_print(ion.Vee_blue_HF, round_to_dec=4)

    blue_tools.table_print(ion.blue_Uc_HF, round_to_dec=4)

    blue_tools.table_print(ion.Uc, round_to_dec=4)

    U_c_error = 100 * (ion.Uc - ion.blue_Uc_HF) / ion.Uc

    blue_tools.table_print(U_c_error, round_to_dec=1)

    blue_tools.table_print(ion.blue_Ec_HF, round_to_dec=4)

    blue_tools.table_print(ion.Ec, round_to_dec=4)

    E_c_error = 100 * (ion.Ec - ion.blue_Ec_HF) / ion.Ec
    blue_tools.table_print(E_c_error, round_to_dec=1, last_in_row=True)

# exact results
print(
    ' Z   Ex(exact)      Vee_blue      Uc_blue       Uc(exact)   Uc(error)   Ec(blue)    Ec(exact)  Ec(error) ')
for i, ion in enumerate(Z_results):
    Z = str(ion.Z)
    blue_tools.table_print(Z)

    blue_tools.table_print(ion.Ex, round_to_dec=4)

    blue_tools.table_print(ion.Vee_blue, round_to_dec=4)

    blue_tools.table_print(ion.blue_Uc, round_to_dec=4)

    blue_tools.table_print(ion.Uc, round_to_dec=4)

    U_c_error = 100 * (ion.Uc - ion.blue_Uc) / ion.Uc

    blue_tools.table_print(U_c_error, round_to_dec=1)

    blue_tools.table_print(ion.blue_Ec, round_to_dec=4)

    blue_tools.table_print(ion.Ec, round_to_dec=4)

    E_c_error = 100 * (ion.Ec - ion.blue_Ec) / ion.Ec
    blue_tools.table_print(E_c_error, round_to_dec=1, last_in_row=True)
