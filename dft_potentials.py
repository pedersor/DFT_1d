import ext_potentials
import numpy as np


def tot_KS_potential(grids, n, v_ext, v_h, v_xc, polarization):
	return v_ext(grids) + v_h(grids=grids, n=n) + v_xc(n, polarization)


def hartree_potential_exp(grids, n, A, k, a):
	N = len(grids)
	dx = np.abs(grids[1] - grids[0])
	v_hartree = np.zeros(N)
	for i in range(N):
		for j in range(N):
			v_hartree[i] += n[j] * (-1) * ext_potentials.exp_hydrogenic(grids[i] - grids[j], A, k, a, Z=1)
	v_hartree *= dx
	return v_hartree


class exchange_correlation_functional(object):

	def __init__(self, grids, A, k):
		self.grids = grids
		self.A = A
		self.k = k
		self.dx = (grids[-1] - grids[0]) / (len(grids) - 1)

	# Exchange-Correlation Potential
	def v_xc_exp(self, n, polarization):
		pi = np.pi

		n_up = polarization[0]
		n_down = polarization[1]
		zeta = polarization[2]

		firstU = self.first(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
		firstP = self.first(n, 180.891, -541.124, 651.615, -356.504, 88.0733, -4.32708, 8)
		secondU = self.second(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
		secondP = self.second(n, 180.891, -541.124, 651.615, -356.504, 88.0733, -4.32708, 8)

		# These are the correct exchange/correlation potentials for the fully polarized case (zeta = 1)
		#v_x = -(self.A/pi) * np.arctan(2*pi*n/self.k)
		#v_c = self.A*n * (n*secondP - 2*firstP) / ((firstP**2)*self.k)

		# These are the correct exchange/correlation potentials for the unpolarized case (zeta = 0)
		#v_x = -(self.A/pi) * np.arctan(pi*n/self.k)
		#v_c = self.A*n * (n*secondU - 2*firstU) / ((firstU**2)*self.k) 

		# These are the result of differentiating the exchange/correlation energy with respect to n_up and n_down explicitly (Attempt 1) 
		#v_x = -(self.A/(2*pi)) * (np.arctan(2*pi*n_up/self.k) + np.arctan(2*pi*n_down/self.k))
		#v_c = (self.A/self.k) * ((((n_down-n_up)**2)*secondP/(firstP**2)) + ((4*n_down*n_up*secondU - 2*firstU*n)/(firstU**2)))

		# These are Attempt 1's UP Potential
		v_x = -(self.A/pi) * np.arctan(2*n_up*pi/self.k)
		v_c = self.A * (2*firstP*(firstU**2)*(n_down - n_up) + (firstU**2)*((n_down - n_up)**2)*secondP - 4*(firstP**2)*n_down*(firstU - n_up*secondU)) / ((firstP**2)*(firstU**2)*self.k)

		# These are the result of differentiating the exchange/correlation energy with respect to n, treating zeta as a constant (Attempt 2)
		#v_x = -(self.A/(2*pi)) * ((1+zeta)*np.arctan(pi*n*(1+zeta)/self.k) + (-1+zeta)*np.arctan(pi*n*(-1+zeta)/self.k))
		#v_c = self.A*n * (-2*firstP*(firstU**2)*(zeta**2) + (firstU**2)*n*secondP*(zeta**2) + (firstP**2)*(2*firstU - n*secondU)*(-1+(zeta**2))) / ((firstP**2)*(firstU**2)*self.k)

		# These are the new DOWN potential from plotting
		#v_x = (self.A/(2*n*pi))*((n+2*n_down-n*zeta)*np.arctan(n*pi*(-1+zeta)/self.k) - (n-2*n_down+n*zeta)*np.arctan(n*pi*(1+zeta)/self.k))
		#v_c = self.A * ((firstP**2)*n*(n*secondU - 2*firstU) + 4*firstP*firstU*(firstU-firstP)*n_down*zeta + n*(firstU*(2*firstP*(firstP-firstU) + firstU*n*secondP) - (firstP**2)*n*secondU)*(zeta**2)) / ((firstP**2)*(firstU**2)*self.k)

		return v_x + v_c

	def first(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
		y = np.pi * n / self.k
		return alpha + beta*(y**(1./2.)) + gamma*y + delta*(y**(3./2.)) + eta*(y**2) + sigma*(y**(5./2.)) + nu*(np.pi*(self.k**2)/self.A)*(y**3)

	def second(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
		return (6*(n**3)*(np.pi**4)*self.k*nu + self.A*np.sqrt(np.pi)*(4*(n**2)*(np.pi**(3./2.))*eta + 2*n*np.sqrt(np.pi)*gamma*self.k + 3*n*np.pi*delta*np.sqrt(n/self.k)*self.k + beta*np.sqrt(n/self.k)*(self.k**2) + 5*n*(np.pi**2)*((n/self.k)**(3./2.))*self.k*sigma)) / (2*self.A*n*(self.k**2))

	# Exchange Energy per Length
	def eps_x(self, n, zeta):
		y = np.pi * n / self.k
		return self.A*self.k*(np.log(1+(y**2)*((1+zeta)**2)) - 2*y*(1+zeta)*np.arctan(y*(1+zeta)) + np.log(1+(y**2)*((-1+zeta)**2)) - 2*y*(-1+zeta)*np.arctan(y*(-1+zeta))) / (4*(np.pi**2))

	# Correlation Energy per Length
	def eps_c(self, n, zeta):
		unpol = self.corrExpression(n, 2, -1.00077, 6.26099, -11.9041, 9.62614, -1.48334, 1)
		pol = self.corrExpression(n, 180.891, -541.124, 651.615, -356.504, 88.0733, -4.32708, 8)
		return unpol + (zeta**2)*(pol - unpol)

	def corrExpression(self, n, alpha, beta, gamma, delta, eta, sigma, nu):
		y = np.pi * n / self.k
		return (-self.A*self.k*(y**2) / (np.pi**2)) / (alpha + beta*(y**(1./2.)) + gamma*y + delta*(y**(3./2.)) + eta*(y**2) + sigma*(y**(5./2.)) + nu*(np.pi*(self.k**2)/self.A)*(y**3))
	
	# Total Exchange Energy
	def E_x(self, n, zeta):
		return self.eps_x(n, zeta).sum() * self.dx

	# Total Correlation Energy
	def E_c(self, n, zeta):
		return self.eps_c(n, zeta).sum() * self.dx