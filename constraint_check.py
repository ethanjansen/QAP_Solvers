############# Get QAP from QAPLIB #####################
import cvxpy as cp
import numpy as np
import pandas as pd
import math as mt
import sys
np.set_printoptions(threshold=sys.maxsize)

file = "QAP_PSD_SolverOut.csv"

# parse data from dat file
val = pd.read_csv(file, sep=",", header=None, nrows=1)[0][0]
df = pd.read_csv(file, sep=",", header=None, skiprows=1)
Y = df.to_numpy(dtype=np.float32)
m = Y.shape[0]
n = int(np.sqrt(m-1))

###################### Setup ##############################

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

############# Constraints ############################
# Checks within precision??
# Check Y is PSD
# Check Y[0][0] == 1
# Diag(Y) == Y[:,0]
# Trace(D*Y) == 0
# b0(Y) == I
# o0(Y) == I

# Y[0][0] == 1
print("Y[0][0] == 1: ", Y[0][0])
print(mt.isclose(Y[0][0], 1, rel_tol=1e-6))

# Diag(Y) == Y[:,0]
print("Diag(Y) == Y[:,0]:")
print(np.isclose(Y[:,0], Y.diagonal, rtol=1e-6))

# Trace(D*Y) == 0
print("Trace(D*Y) == 0:")
print(mt.isclose(np.trace(np.matmul(D, Y)), 0, rel_tol=1e-6))

# b0(Y) == I
print("b0(Y) == I:")
print(np.isclose(b0(Y), I, rtol=1e-6))

# o0(Y) == I
print("o0(Y) == I:")
print(np.isclose(o0(Y), I, rtol=1e-6))

# Y is PSD
print("Y is PSD:")
try:
    np.linalg.cholesky(Y)
except LinAlgError:
    print(False)
else:
    print(True)
finally:
    print("Done")