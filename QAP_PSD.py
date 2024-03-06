############# Get QAP from QAPLIB #####################
import cvxpy as cp
import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

link = "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/esc16a.dat"
# doesnt work with parsing: link = "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/chr12a.dat"

# parse data from dat file
n = pd.read_csv(link, sep=" ", header=None, nrows=1)[0][0]
m = n**2 + 1 # final size after expanding to R^{nxn+1}
l = m - (2*n) + 1
df = pd.read_csv(link, sep=" ", header=None, skiprows=1)
A = df.iloc[:n].to_numpy(dtype=np.float64)
B = df.iloc[n:].to_numpy(dtype=np.float64)
# C = 0 for pure QAP

############# PSD Variable ############################
#Y = cp.Variable((m,m), PSD=True)
Z = cp.Variable((l,l), PSD=True)
############# Objective Function ######################
# Lq = [[0, 0], [0, B kron A]]
# Y = [[1, x^T], [x, xx^T]] -- will find with PSD solver
Lq = np.kron(B, A)
Lq = np.r_[[[0]*(m-1)], Lq]
Lq = np.c_[[0]*m, Lq] # Lq

###################### Constraints ############################
I = np.identity(n) # identity
E = np.ones(shape=(n,n)) # ones matrix
D = np.kron(I, E) + np.kron(E, I)
D = np.r_[[[-2]*(m-1)], D]
D = np.c_[[-2]*m, D]
D[0,0] = 2*n # D

V = np.r_[np.identity(n-1), [-1*np.ones(n-1)]] # V
Vhat = np.kron(V, V)
Vhat = np.r_[[np.zeros(l-1)],Vhat]
Vhat = np.c_[np.ones(m)/n,Vhat]
Vhat[0][0] = 1

# actual constraints
constraints = [Y[0][0] == 1]
constraints += [cp.diag(Y) - Y[:,0] == np.zeros(m)]
constraints += [cp.sum([Y[i:i+n, i:i+n] for i in range(1,m,n)]) == I]
constraints += [cp.sum([Y[i:m:n, i:m:n] for i in range(1,n+1)]) == I]
constraints += [cp.trace(D @ Y) == 0]
constraints += [Y == Vhat @ Z @ cp.trace(Vhat)]

################# Objective & Solve ####################
PSDprob = cp.Problem(cp.Minimize(cp.trace(Lq @ Y)), constraints) # minimize tr(LqY) subject to the constraints
PSDprob.solve(verbose=True)

# Print results
print("The optimal value is ", PSDprob.value)
#print("A solution Y is")
#print(Y.value)

y = 1.5
f = open("QAP_PSD_SolverOut.csv", "w")
f.write(str(PSDprob.value)+'\n')
f.close()

df = pd.DataFrame(Y.value)
df.to_csv("QAP_PSD_SolverOut.csv", mode='a', header=False, index=False)
