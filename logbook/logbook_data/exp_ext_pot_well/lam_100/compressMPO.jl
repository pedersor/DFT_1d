#!/Applications/Julia-0.5.app/Contents/Resources/julia/bin/julia
using LinearAlgebra

function printnonzero(f::IOStream, mat::Array{Float64,2})
	nr,nc = size(mat)
	numterm = 0
	for i=1:nr, j=1:nc
    	mat[i,j] != 0.0 && (numterm += 1)
  	end
  	write(f,"$numterm\n")
  	for i=1:nr, j=1:nc
    	mat[i,j] != 0.0 && write(f,"$i $j $(mat[i,j])\n")
  	end
end

parseint(s) = parse(Int64,s)
parsefloat(s) = parse(Float64,s)

if length(ARGS) < 1
  println("Usage: julia compressMPO.jl integral_file")
end

fname = ARGS[1]
fi = open(fname,"r")
N = 1
p = split(readline(fi))
N = parseint(p[1]) # 2 times number of nuclei in the system

V = zeros(N,N)
while !eof(fi)
	p = split(readline(fi))
	V[parseint(p[1]),parseint(p[2])] = parsefloat(p[3])
end
#starting = true
#while true
#	p = split(readline(fi))
#	parseint(p[1]) > 1 && (starting = false)
#	if starting == false && parseint(p[1]) == 1 
#		V[parseint(p[1]),parseint(p[2])] = parsefloat(p[3])
#		break
#  	end
#end
#while true 
#	p = split(readline(fi))
#	length(p) != 3 && break
#	V[parseint(p[1]),parseint(p[2])] = parsefloat(p[3])
#end

#println("V = "); printnonzero(V); exit(0)

cutoff = 1E-12

U = Array{Any}(undef,N)
X = Array{Any}(undef,N)
W = Array{Any}(undef,N)
row1 = V[1,:]
U[1] = ones(1,1)
W[1] = reshape(row1,1,N)
maxM = 0
maxDiff = 0
for i=2:(N-1)
	Waug = vcat(W[i-1][:,2:end],V[i:i,i:N])
	Nu = size(U[i-1],2)
	Uaug = zeros(i,Nu+1)
	Uaug[1:i-1,1:Nu] = U[i-1]
	Uaug[i,Nu+1] = 1.0
	(UU,S,VV) = svd(Waug)
	M = findlast(x -> x > cutoff*S[1],S)
	global maxM
	maxM = max(maxM,M)
	W[i] = diagm( 0 => S[1:M]) * VV[:,1:M]'
	X[i] = UU[:,1:M]
	U[i] = Uaug * X[i]
	Vblock = V[1:i,i:N]
	dif = norm(Vblock-U[i] * W[i])
	global maxDiff
	dif > maxDiff && (maxDiff = dif)
end
@show maxM
@show maxDiff


# Product of all MPO mats from the left up to l gives a vector:
# el 1:  1 1 1 1 1
# el 2:  Hleft
# el j= 3 to 3+la:  sum_i n_i U[l]_ij

#println("\nResult")

f = open("Vcompressed","w")

write(f,"$N\n")

for i=1:N
	nr = 0
	nc = 0
	if i==1
		nr = 1
		nc = 3
	elseif i == N
    	nr = 2+size(U[i-1],2)
    	nc = 1
  	else
    	nr = 2+size(U[i-1],2)
    	nc = 2+size(U[i],2)
  	end
  	write(f,"$i $(nr) $(nc)\n") # site, nrow, ncol

  	write(f,"Id\n")
  	mat = zeros(nr,nc)
  	if i < N
    	mat[1,1] = 1
  	end
  	if 1 < i < N
    	mat[2,2] = 1
    	mat[3:nr,3:nc] = X[i][1:nr-2,1:nc-2]
  	elseif i == N
    	mat[2,1] = 1
  	end
  	printnonzero(f,mat)

	write(f,"Nupdn\n")
  	mat = zeros(nr,nc)
  	if i < N
    	mat[1,2] = V[i,i]
  	elseif i == N
    	mat[1,1] = V[i,i]
  	end
  	printnonzero(f,mat)

	write(f,"Ntot\n")
  	mat = zeros(nr,nc)
  	if 1 < i < N
    	mat[3:nr,2] = W[i-1][:,2]
  	elseif i == N
    	mat[3:nr,1] = W[i-1][:,2]
  	end
  	if i < N
    	mat[1,3:nc] = U[i][i,:]
  	end
  	printnonzero(f,mat)
end

close(f)
