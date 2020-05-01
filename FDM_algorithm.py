#algorithm to produce arbitrary FDM formulas

#split a region into even grids
#N grids, N+1 points
#shape examples:
#1.increasing: [0, 1, 2, 3..]
#2.decreasing: [10, 9, 8, 7..]
#3.centered:[0, 1, -1, 2, -2..]
#           [1/2, -1/2, 3/2, -3/2..]
def grids(N, min, max, shape):
    delta = (max - min)/N
    if shape == "increasing":
        return [min+delta*i for i in range(N+1)]
    elif shape == "decreasing":
        return [max-delta*i for i in range(N+1)]
    elif shape == "centered":
        mid = (max + min)/2
        if N%2 == 0:
            grids = [mid]
            for i in range(1, N+1):
                grids.append(grids[i-1]-i*delta*(-1)**i)
            return grids
        else:
            grids = [mid + delta/2]
            for i in range(1, N+1):
                grids.append(grids[i-1]+i*delta*(-1)**i)
            return grids
    else:
        return NotImplemented


#algorithm from the paper
#https://doi.org/10.1090/S0025-5718-1988-0935077-0
def generate_sigma(M, N, x0, a):
    
    #initialize to be all zeros
    sigma = []
    for i in range(M+1):
        sigma.append([])
        for j in range(N+1):
            sigma[i].append([])
            for k in range(N+1):
                sigma[i][j].append(0)
    
    #algorithm from paper
    sigma[0][0][0] = 1
    c1 = 1
    for n in range(1, N+1):
        c2 = 1
        for v in range(0, n):
            c3 = a[n] - a[v]
            c2 = c2*c3
            for m in range(0, min(n, M)+1):
                sigma[m][n][v] = ((a[n]-x0)*sigma[m][n-1][v] - (0 if m == 0 else m*sigma[m-1][n-1][v]))/c3
        for m in range(0, min(n, M)+1):       
            sigma[m][n][n] = (c1/c2)*((0 if m == 0 else m*sigma[m-1][n-1][n-1])-(a[n-1]-x0)*sigma[m][n-1][n-1])        
        c1 = c2
        
    return sigma

#print results
#m = order of derivative
#n-m+1 = order of accuracy
#not specifying m and n would print all results
def print_results(sigma, x0, a, m = None, n = None):
    for i in (list(m) if m else range(len(sigma))):
        print(f"m = {i}:")
        print("a =", a)
        print("around x0 =", x0)
        for j in (list(n) if n else range(len(sigma[i]))):
            print(f"n = {j} / accuracy = {j-i+1}:", sigma[i][j])
        print()
        

if __name__ == "__main__":
    M = 4 #largest order of derivatives
    N = 10 #N grids, N+1 points
    x0 = 0 #central point
    a = grids(N, 0, 10, "increasing") #grids
    sigma = generate_sigma(M, N, x0, a)
    print_results(sigma, x0, a, [2], [3, 5])