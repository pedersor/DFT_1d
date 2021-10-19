using Printf

N = 513 #number of grid points in x direction
delta = 0.08 #grid spacing
R = 25
Nc = 257 # position of center site of the grid
lam = 1300
Nel = 2
X_array = [] # positions of nuclei


#potential parameters
A = 1.071295
kpiv = 2.385345

Vpp = A*exp(-R*delta/kpiv) #change for num_el > 2
@show Vpp

#hopping coefficients t
te =  Float64[30,-16,1]/24 / delta^2

function potpe(i::Int64)
	potpeval = 0

	potpeval = 0.5*(0.25)*((i-Nc)*(i-Nc)*delta*delta)
	
	return potpeval
end

f = open("Ham1c","w")
	write(f,"$N\n")#include Nup Ndn later
	write(f,"$Vpp\n")
	for i = 1:N
		for j = 1:N
			if i == j
				@printf(f,"%d %d %0.20f\n",i,j,te[1]+potpe(i))
			elseif i-j == 1 || j-i == 1
				@printf(f,"%d %d %0.20f\n",i,j,te[2])
			elseif i-j == 2 || j-i == 2
				@printf(f,"%d %d %0.20f\n",i,j,te[3])
			end
		end
	end

#interaction potential between electrons
function pot(i::Int64,j::Int64)
	return lam*A*exp(-abs(i-j)*delta/kpiv)
end

open("Vuncomp","w") do f
	write(f,"$N\n")
	for i=1:N
		for j=1:N
			V = pot(i,j)
			@printf(f,"%d %d %0.20f\n",i,j,V)
		end
	end
end

close(f)
