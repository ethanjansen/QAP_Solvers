############# Setup LAP #####################
import cvxpy as cp
import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

link = "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/esc16a.dat"
# doesnt work with parsing: link = "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/chr12a.dat"

# parse data from dat file
n = pd.read_csv(link, sep=" ", header=None, nrows=1)[0][0]
df = pd.read_csv(link, sep=" ", header=None, skiprows=1)
C = df.to_numpy(dtype=np.float64)

############# PSD Variable ############################
X = cp.Variable((n,n))

###################### Constraints ############################
eT = np.ones(n)

constraints = [X >> 0]
constraints += [e @ X == e]
constraints += [e @ X.T == e]

################# Objective & Solve ####################
PSDprob = cp.Problem(cp.Minimize(cp.trace(C @ X.T)), constraints)
PSDprob.solve(verbose=True)

# Print results
print("The optimal value is ", PSDprob.value)
#print("A solution Y is")
#print(Y.value)

f = open("LAP_PSD_SolverOut.csv", "w")
f.write(str(PSDprob.value)+'\n')
f.close()

print("Saving X")
df = pd.DataFrame(X.value)
df.to_csv("LAP_PSD_SolverOut.csv", mode='a', header=False, index=False)
