"""Default constants used in this library.

Exponential Coulomb interaction.

v(x) = amplitude * exp(-abs(x) * kappa)

See also ext_potentials.exp_hydrogenic. Further details in:

Thomas E Baker, E Miles Stoudenmire, Lucas O Wagner, Kieron Burke,
and  Steven  R  White. One-dimensional mimicking of electronic structure:
The case for exponentials. Physical Review B,91(23):235141, 2015.
"""

EXPONENTIAL_COULOMB_AMPLITUDE = 1.071295
EXPONENTIAL_COULOMB_KAPPA = 1 / 2.385345
