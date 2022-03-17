
###############################################################################
#####                             MAIN PROGRAM                            #####
###############################################################################
import sys
sys.path.append("./libs_v0")

# Python libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix as csr

# My libs
import pcg
from laplace import *


nSide = 5
n = nSide
A = laplace1D(nSide)

b = np.ones((n,1))
x0 = np.ones((n,1))

solver = pcg.pcg(n,A,b)
solver.solve(x0)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.linspace(0,1,n),solver.sol)


print('Solution = ', solver.sol.transpose())
print('Residual = ', np.abs(A@solver.sol-b).transpose())



plt.show()

