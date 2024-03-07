############# Get SOL from File #####################
import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

file = "LAP_PSD_SolverOut.csv"

# parse data from dat file
val = pd.read_csv(file, sep=",", header=None, nrows=1)[0][0]
df = pd.read_csv(file, sep=",", header=None, skiprows=1)
X = df.to_numpy(dtype=np.float32).astype(np.int32) # get int32 (to float32 first to round) for exact values
n = X.shape[0]

################# Reconstruct X #####################
print(X)