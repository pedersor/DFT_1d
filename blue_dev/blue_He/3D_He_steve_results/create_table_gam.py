import numpy as np
import blue_tools

# exact quantities (using s=0.7, c=0.2 basis)
U_H_exact = 2.04904
E_x_exact = -1.02452
V_ee_exact = 0.94587
U_xc_exact = V_ee_exact - U_H_exact
U_c_exact = U_xc_exact - E_x_exact

gammas, V_ee_blue = blue_tools.txt_file_to_array('V_ee_blue.dat', header=True)

for i, V_ee_blue_gam in enumerate(V_ee_blue):
    method = 'blue ($\gamma =' + str(gammas[i]) + '$)'
    blue_tools.table_print(method)

    U_xc_blue_gam = V_ee_blue_gam - U_H_exact
    blue_tools.table_print(U_xc_blue_gam, round_to_dec=4)

    U_c_blue_gam = U_xc_blue_gam - E_x_exact
    blue_tools.table_print(U_c_blue_gam, round_to_dec=4)

    U_c_error = 100 * (U_c_exact - U_c_blue_gam) / (U_c_exact)
    blue_tools.table_print(U_c_error, round_to_dec=1, last_in_row=True)

blue_tools.table_print('exact')
blue_tools.table_print(U_xc_exact, round_to_dec=4)
blue_tools.table_print(U_c_exact, round_to_dec=4)
blue_tools.table_print(0.0, round_to_dec=1, last_in_row=True)
