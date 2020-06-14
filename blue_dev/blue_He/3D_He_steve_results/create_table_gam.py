import numpy as np
import blue_tools


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
        self.Tc = self.get_T_c(self.Uc)
        self.Ec = self.Uc + self.Tc

    def get_blue_results(self, Vee_blue):
        self.Vee_blue = Vee_blue
        self.blue_Uxc = Vee_blue - self.U
        self.blue_Uc = self.blue_Uxc - self.Ex
        self.blue_Tc = self.get_T_c(self.blue_Uc)
        self.blue_Ec = self.blue_Uc + self.blue_Tc
        return

    def get_T_c(self, Uc):
        T_c = 0.5 * (-2 * self.Ts - self.V_ext - self.U - self.Ex - Uc)
        return T_c


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

Z_results = []
for i, ion in enumerate(Z):
    Z_Quantities = Quantities(ion, Etot[i], Ex_plus_U[i], Ts[i], Vee[i])
    Z_Quantities.get_blue_results(Vee_blue[i])
    Z_results.append(Z_Quantities)

# cyrus = [0.499869, 2.867082, 7.240085, 13.614084, 32.363072]
# print('Z  |  T_s  |  T_s(Cyrus)   |    T_s(Cyrus) - T_s')


for i, ion in enumerate(Z_results):
    Z = str(ion.Z)
    blue_tools.table_print(Z)

    blue_tools.table_print(ion.blue_Uc, round_to_dec=4)

    blue_tools.table_print(ion.Uc, round_to_dec=4)

    blue_tools.table_print(ion.blue_Ec, round_to_dec=4)

    blue_tools.table_print(ion.Ec, round_to_dec=4, last_in_row=True)

