############# Get SOL from File #####################
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
Y = df.to_numpy(dtype=np.float64)
m = Y.shape[0]
n = int(np.sqrt(m-1))

################# Reconstruct X #####################
X = np.reshape(Y[0,1:], (n,n)).T
print(X)
