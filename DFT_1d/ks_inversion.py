"""
Kohn-Sham Inversion
###################


.. todo::

    * Needs code documentation
    * Needs more specific name (since there are many inversions)
    * Needs ref
    * RJM: Re-format ah-la KS-pies?
    * Who was original author of this code?

"""
import logging

import numpy as np

from DFT_1d.non_interacting_solver import EigenSolver
from DFT_1d.utils import IntegralTool, DerivativeTool
from DFT_1d import constants


# https://link.springer.com/article/10.1007%2Fs00214-018-2209-0
class KSInversion:

  def __init__(
      self,
      full_grids,
      v_Z_fn,
      full_target_density,
      truncation,
      total_density=None,
      occupation_per_state=2,
      n_point_stencil=5,
      init_v_XC_fn=lambda x: 0 * x,
      Solver=EigenSolver,
      num_electrons=1,
      mixing_param=0,
      tol=1e-5,
      max_iters=50,
      verbose=False,
  ):

    # get truncation based on the tolerance
    self.truncation = truncation

    # number of electrons
    self.num_electrons = num_electrons
    # electron occupation per state
    self.occ_per_state = occupation_per_state

    # full grids/density and truncated grids/density
    self.f_grids = full_grids
    self.num_f_grids = len(self.f_grids)
    self.t_grids = self.f_grids[truncation:self.num_f_grids - truncation]
    self.num_t_grids = len(self.t_grids)
    self.dx = (self.t_grids[-1] - self.t_grids[0]) / (self.num_t_grids - 1)
    self.f_tar_density = full_target_density
    self.tar_density = self.f_tar_density[truncation:self.num_f_grids -
                                          truncation]
    if total_density is None:
      self.total_density = self.f_tar_density
    else:
      self.total_density = total_density

    # pass truncated versions to tools
    self.t_derivative_tool = DerivativeTool(self.t_grids)
    self.f_derivative_tool = DerivativeTool(self.f_grids)
    self.f_integral_tool = IntegralTool(self.f_grids)

    # initialize full and truncated potentials
    self.f_v_Z = v_Z_fn(self.f_grids)
    self.f_v_H = self.f_integral_tool.hartree_matrix.dot(self.total_density)
    self.t_v_W_tar = self._get_v_W(self.tar_density)
    self.f_v_XC = init_v_XC_fn(self.f_grids)
    self.t_v_XC = self.f_v_XC[truncation:self.num_f_grids - truncation]

    # set tolerance and mixing param
    self.tol = tol
    self.mixing_param = mixing_param

    # initialize solver
    self.n_point_stencil = n_point_stencil
    self.solver = Solver(self.f_grids,
                         n_point_stencil=n_point_stencil,
                         num_electrons=num_electrons)

    # initialize density and kinetic energy
    self.previous_f_density = None
    self.f_density = None
    self.t_density = None
    self.previous_KE = None
    self.KE = None

    # whether has solved full v_XC
    self._solved = False

    # set max number of iterations
    self.max_iters = max_iters

    # print debug information
    if verbose:
      logging.basicConfig(level=logging.DEBUG)

  def get_v_s(self):
    f_v_Z = self.f_v_Z
    f_v_H = self.f_v_H
    f_v_XC = self.f_v_XC
    return f_v_Z + f_v_H + f_v_XC

  def _get_v_W(self, density):
    d1_mat = self.t_derivative_tool.d1_mat
    d2_mat = self.t_derivative_tool.d2_mat
    v_W_n1 = ((d1_mat.dot(density) / density)**2) / 8
    v_W_n2 = (d2_mat.dot(density) / density) / 4
    v_W_n = v_W_n1 - v_W_n2
    return v_W_n

  def _update_density(self):
    # d2_mat = self.f_derivative_tool.d2_mat
    potential_fn = lambda _: self.get_v_s()
    self.solver.update_potential(potential_fn)
    self.solver.solve_ground_state(self.occ_per_state)
    self.previous_KE = self.KE
    self.KE = self.solver.kinetic_energy
    self.previous_f_density = self.f_density
    if self.previous_f_density is None:
      self.f_density = self.solver.density
    else:
      self.f_density = self.solver.density * (
          1 - self.mixing_param) + self.previous_f_density * self.mixing_param
    self.t_density = self.f_density[self.truncation:self.num_f_grids -
                                    self.truncation]

  def _update_v_XC(self, x0, ignored_points, f):
    t_v_W_n = self._get_v_W(self.t_density)
    self.t_v_XC = self.t_v_XC - self.t_v_W_tar + t_v_W_n
    if self.truncation == 0:
      self.f_v_XC = self.t_v_XC
    else:
      self.append_exp(x0, ignored_points, f)

  def _get_tar_density_cost(self):
    # cost betwen target_density and density
    s_v = self.f_integral_tool.simpson_vector
    d_n = self.f_density
    d_tar = self.f_tar_density
    return ((s_v.dot((d_n - d_tar)**2))**0.5) / self.num_electrons

  def _get_f_density_cost(self):
    if self.previous_f_density is not None:
      s_v = self.f_integral_tool.simpson_vector
      d_n = self.f_density
      d_p = self.previous_f_density
      return ((s_v.dot(abs(d_n - d_p)**2))**0.5) / self.num_electrons
    else:
      return None

  def _get_KE_cost(self):
    if self.previous_KE is not None:
      return abs(self.previous_KE - self.KE)
    else:
      return None

  def _append_exp_cfexp(self, x0, ignored_points):
    '''Central point formula at the end, but including 
        exponential decay inside the FDM formula.'''
    if self.n_point_stencil == 5:
      d1_h = np.array([1 / 12, -2 / 3, 0., 2 / 3, -1 / 12])
    elif self.n_point_stencil == 3:
      d1_h = np.array([0., -0.5, 0., 0.5, 0.])

    ip = ignored_points
    # k_value
    k = constants.EXPONENTIAL_COULOMB_KAPPA

    # grid value at right end
    xr = self.t_grids[-ip - 1]
    # use right end value to calculate the middle offset
    vr = self.t_v_XC[-ip - 1]

    Ar0 = np.exp(-k * abs(xr - x0))
    Ar1 = np.exp(-k * abs(xr - x0 + self.dx))
    Ar2 = np.exp(-k * abs(xr - x0 + 2 * self.dx))

    Cr_numerator = d1_h.dot(
        np.append(self.t_v_XC[-ip - 3:-ip - 1], [vr, vr, vr]))
    Cr_denominator = d1_h.dot(np.array([0, 0, 0, Ar0 - Ar1, Ar0 - Ar2
                                       ])) - self.dx * k * Ar0

    Cr = Cr_numerator / Cr_denominator
    offset_m = Cr * Ar0 - vr

    exp_r = lambda x: Cr * np.exp(-k * abs(x - x0))

    # adjust v_XC with offset_m, append right exponential
    self.r_grids = self.f_grids[self.num_f_grids - self.truncation - ip:]
    self.f_v_XC = np.append(self.t_v_XC[ip:-ip] + offset_m, exp_r(self.r_grids))

    # grid value at left end
    xl = self.t_grids[ip]
    # use middle-left end value to calculate the left offset
    vl = self.t_v_XC[ip]

    Al0 = np.exp(-k * abs(xl - x0))
    Al1 = np.exp(-k * abs(xl - x0 - self.dx))
    Al2 = np.exp(-k * abs(xl - x0 - 2 * self.dx))

    Cl_numerator = d1_h.dot(np.append([vl, vl, vl], self.t_v_XC[ip + 1:ip + 3]))
    Cl_denominator = d1_h.dot(np.array([Al0 - Al2, Al0 - Al1, 0, 0, 0
                                       ])) + self.dx * k * Al0

    Cl = Cl_numerator / Cl_denominator
    offset_l = vl + offset_m - Cl * Al0

    exp_l = lambda x: Cl * np.exp(-k * abs(x - x0))

    # append left exponential
    self.l_grids = self.f_grids[:self.truncation + ip]
    self.f_v_XC = np.append(exp_l(self.l_grids) + offset_l, self.f_v_XC)

  def _append_exp_cf(self, x0, ignored_points):
    # append C*exp[-k|x-x0|] to both ends with central formula

    ip = ignored_points
    if ip < 3:
      raise ValueError(f'ignored_points should be >= 3, got {ignored_points}.')

    # set up central-point derivative formula
    if self.n_point_stencil == 5:
      d1 = np.array([1 / 12, -2 / 3, 0., 2 / 3, -1 / 12]) / self.dx
    elif self.n_point_stencil == 3:
      d1 = np.array([0., -0.5, 0., 0.5, 0.]) / self.dx

    # k_value
    k = constants.EXPONENTIAL_COULOMB_KAPPA

    # grid value at right end
    xr = self.t_grids[-ip - 1]
    # right end derivative
    dr = d1.dot(self.t_v_XC[-ip - 3:-ip + 2])
    # A for right end
    Cr = dr / (-k * np.exp(-k * abs(xr)))
    exp_r = lambda x: Cr * np.exp(-k * abs(x - x0))

    # use right end value to calculate the middle offset
    vr = self.t_v_XC[-ip - 1]
    offset_m = exp_r(xr) - vr

    # adjust v_XC with offset_m, append right exponential
    self.r_grids = self.f_grids[self.num_f_grids - self.truncation - ip:]
    self.f_v_XC = np.append(self.t_v_XC[ip:-ip] + offset_m, exp_r(self.r_grids))

    # grid value at left end
    xl = self.t_grids[ip]
    # left end derivative
    dl = d1.dot(self.t_v_XC[ip - 2:ip + 3])
    # C for left end
    Cl = dl / (-k * np.exp(-k * abs(xl)))
    exp_l = lambda x: -Cl * np.exp(-k * abs(x - x0))

    # use middle-left end value to calculate the left offset
    vl = self.t_v_XC[ip] + offset_m
    offset_l = vl - exp_l(xl)

    # append left exponential
    self.l_grids = self.f_grids[:self.truncation + ip]
    self.f_v_XC = np.append(exp_l(self.l_grids) + offset_l, self.f_v_XC)

  def _append_exp_endf(self, x0, ignored_points):
    # append C*exp[-k|x-x0|] to both ends with end formula

    ip = ignored_points

    # set up end-point derivative formula
    if self.n_point_stencil == 5:
      d1_l = np.array([-25 / 12, 4., -3., 4 / 3, -1 / 4]) / self.dx
    elif self.n_point_stencil == 3:
      d1_l = np.array([-3 / 2, 2., -1 / 2, 0, 0]) / self.dx
    d1_r = -d1_l[::-1]

    # k_value
    k = constants.EXPONENTIAL_COULOMB_KAPPA

    # grid value at right end
    xr = self.t_grids[-ip - 1]
    # right end derivative
    dr = d1_r.dot(self.t_v_XC[-ip - 5:-ip])
    # A for right end
    Cr = dr / (-k * np.exp(-k * abs(xr)))
    exp_r = lambda x: Cr * np.exp(-k * abs(x - x0))

    # use right end value to calculate the middle offset
    vr = self.t_v_XC[-ip - 1]
    offset_m = exp_r(xr) - vr

    # adjust v_XC with offset_m, append right exponential
    self.r_grids = self.f_grids[self.num_f_grids - self.truncation - ip:]
    self.f_v_XC = np.append(self.t_v_XC[ip:-ip] + offset_m, exp_r(self.r_grids))

    # grid value at left end
    xl = self.t_grids[ip]
    # left end derivative
    dl = d1_l.dot(self.t_v_XC[ip:ip + 5])
    # C for left end
    Cl = dl / (-k * np.exp(-k * abs(xl)))
    exp_l = lambda x: -Cl * np.exp(-k * abs(x - x0))

    # use middle-left end value to calculate the left offset
    vl = self.t_v_XC[ip] + offset_m
    offset_l = vl - exp_l(xl)

    # append left exponential
    self.l_grids = self.f_grids[:self.truncation + ip]
    self.f_v_XC = np.append(exp_l(self.l_grids) + offset_l, self.f_v_XC)

  def append_exp(self, x0=0, ignored_points=0, f='cfexp'):
    if f == 'cf':
      self._append_exp_cf(x0, ignored_points)
    elif f == 'endf':
      self._append_exp_endf(x0, ignored_points)
    elif f == 'cfexp':
      self._append_exp_cfexp(x0, ignored_points)

  def solve_v_XC(self, x0=0, ignored_points=0, f='cfexp', truncated=False):

    self.step_list = []
    self.cost_list = [[], [], []]
    f_density_cost = None
    tar_density_cost = 1
    for step in range(self.max_iters):
      self._update_density()
      tar_density_cost = self._get_tar_density_cost()
      f_density_cost = self._get_f_density_cost()
      KE_cost = self._get_KE_cost()
      self.step_list.append(step)
      self.cost_list[0].append(tar_density_cost)
      self.cost_list[1].append(f_density_cost)
      self.cost_list[2].append(KE_cost)
      logging.debug(
          f'Step {step}: current target density cost {tar_density_cost}')
      logging.debug(f'Step {step}: current full density cost {f_density_cost}')
      logging.debug(f'Step {step}: current KE cost {KE_cost}')
      logging.debug('')
      if f_density_cost != None and KE_cost != None:
        if (((tar_density_cost <= self.tol or not truncated) and
             KE_cost <= self.tol) or step > self.max_iters):
          break

      self._update_v_XC(x0, ignored_points, f)
    self._solved = True

    return self


# Kohn-Sham inversion with two iterations: one with truncated region
# and the other with full region
def two_iter_KS_inversion(
    f_grids,
    f_v_Z_fn,
    f_tar_density,
    num_electrons,
    total_density=None,
    occupation_per_state=2,
    init_v_xc=None,
    n_point_stencil=5,
    mixing_param=0,
    tol=0.00001,
    t_tol=0.0001,
    max_iters=50,
):

  if init_v_xc is None:
    init_v_xc = np.zeros_like(f_grids)

  def find_truncation(density, t_tol):
    for t in range(len(density)):
      if density[t] > t_tol:
        return t

  def set_up_truncation(f_grids, f_v_Z_fn, density, init_v_xc, truncation):
    t_grids = f_grids[truncation:-truncation]
    t_v_Z_fn = lambda grids: f_v_Z_fn(grids)[truncation:-truncation]
    t_density = density[truncation:-truncation]
    t_init_v_xc = init_v_xc[truncation:-truncation]
    return t_grids, t_v_Z_fn, t_density, t_init_v_xc

  truncation = find_truncation(f_tar_density, t_tol)
  t_grids, t_v_Z_fn, t_tar_density, t_init_v_xc = set_up_truncation(
      f_grids, f_v_Z_fn, f_tar_density, init_v_xc, truncation)

  if total_density is not None:
    _, _, t_total_density, _ = set_up_truncation(f_grids, f_v_Z_fn,
                                                 total_density, t_init_v_xc,
                                                 truncation)
  else:
    t_total_density = None

  logging.debug(f'Running truncation {truncation}')

  t_ksi = KSInversion(
      t_grids,
      t_v_Z_fn,
      t_tar_density,
      truncation=0,
      n_point_stencil=n_point_stencil,
      tol=tol,
      num_electrons=num_electrons,
      total_density=t_total_density,
      init_v_XC_fn=lambda _: t_init_v_xc,
      occupation_per_state=occupation_per_state,
      mixing_param=mixing_param,
      max_iters=max_iters,
  )

  t_ksi.solve_v_XC(f='cfexp')

  old_truncation = truncation
  truncation += find_truncation(t_ksi.f_density, t_tol)
  logging.debug(f'truncation updated from {old_truncation} to {truncation}.')
  t_grids, t_v_Z_fn, t_tar_density, t_init_v_xc = set_up_truncation(
      f_grids, f_v_Z_fn, f_tar_density, t_init_v_xc, truncation)

  ksi = KSInversion(
      f_grids,
      f_v_Z_fn,
      f_tar_density,
      truncation=truncation,
      n_point_stencil=n_point_stencil,
      tol=tol,
      num_electrons=num_electrons,
      total_density=total_density,
      init_v_XC_fn=lambda _: init_v_xc,
      occupation_per_state=occupation_per_state,
      mixing_param=mixing_param,
      max_iters=max_iters,
  )

  delta = truncation - old_truncation
  ksi.t_v_XC = t_ksi.t_v_XC[delta:-delta]

  ksi.append_exp(x0=0, ignored_points=6, f='cfexp')
  ksi.solve_v_XC(ignored_points=6, f='cfexp', truncated=True)

  logging.debug('done!')
  return ksi
