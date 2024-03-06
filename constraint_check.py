############# Get QAP from QAPLIB #####################
import cvxpy as cp
import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

link = "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/esc16a.dat"

# parse data from dat file
n = pd.read_csv(link, sep=" ", header=None, nrows=1)[0][0]
m = n**2 + 1 # final size after expanding to R^{nxn+1}
df = pd.read_csv(link, sep=" ", header=None, skiprows=1)
A = df.iloc[:n].to_numpy(dtype=np.float32)
B = df.iloc[n:].to_numpy(dtype=np.float32)
# C = 0 for pure QAP

############# PSD Variable ############################
Y = cp.Variable((m,m), PSD=True)

############# Constraints ############################
I = np.identity(n) # identity
E = np.ones(shape=(n,n)) # ones matrix
D = np.kron(I, E) + np.kron(E, I)
D = np.r_[[[-2]*(m-1)], D]
D = np.c_[[-2]*m, D]
D[0,0] = 2*n # D

# block-0-diagonal
def b0(Y):
    m = Y.shape[0] # assuming square
    n = int(np.sqrt(m-1))
    total = np.zeros(shape=(n,n), dtype=np.float32)
    for i in range(1,m,n):
        total += Y[i:i+n, i:i+n]
    return(total)

# off-0-diagonal
def o0(Y):
    m = Y.shape[0] # assuming square
    n = int(np.sqrt(m-1))
    total = np.zeros(shape=(n,n), dtype=np.float32)
    for i in range(1,n+1):
        total += Y[i:m:n, i:m:n]
    return(total)

# actual constraints
constraints = [Y[0][0] == 1]
constraints += [cp.diag(Y) - Y[:,0] == np.zeros(m)]
constraints += [cp.sum([Y[i:i+n, i:i+n] for i in range(1,m,n)]) == I]
constraints += [cp.sum([Y[i:m:n, i:m:n] for i in range(1,n+1)]) == I]
constraints += [cp.trace(D @ Y) == 0]

################# Objective & Solve ####################
PSDprob = cp.Problem(cp.Minimize(cp.trace(Lq @ Y)), constraints) # minimize tr(LqY) subject to the constraints
PSDprob.solve(verbose=True)

# Print result.
print("The optimal value is ", PSDprob.value)
print("A solution Y is")
print(Y.value)