import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

y = 1.5
f = open("test.txt", "w")
f.write(str(y)+'\n')
f.close()

z = np.ones(shape=(3,3))
df = pd.DataFrame(z, )
df.to_csv("test.txt", mode='a', header=False, index=False)