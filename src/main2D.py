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
import mesh
import bddc


nSide = 11

# Matrix
n = nSide*nSide
A = laplace2D(nSide)
b = np.ones((n,1))
x0 = np.ones((n,1))



# Mesh with 4 partitions
Ind = np.arange(0,nSide).reshape((nSide,1))*nSide + np.arange(0,nSide).reshape((1,nSide))

nodeMap = []
partition = np.zeros(n,dtype=int)
for k in range(n):
    l = []
    i = k//nSide
    j = k%nSide
    if (i != 0      ): l.append((i-1)*nSide+j)
    if (j != 0      ): l.append(i*nSide+j-1)
    if (j != nSide-1): l.append(i*nSide+j+1)
    if (i != nSide-1): l.append((i+1)*nSide+j)
    nodeMap.append(l)

    if   (i < nSide//2 and j < nSide//2): partition[k] = 0
    elif (i < nSide//2 and j > nSide//2): partition[k] = 1
    elif (i > nSide//2 and j < nSide//2): partition[k] = 2
    elif (i > nSide//2 and j > nSide//2): partition[k] = 3
    else: partition[k] = -1



M = mesh.mesh(nodeMap,partition)
print(' > Mesh Structure: ')
print(M.nodeMap)
print(M.parts)
print('')

print(' > List of objects: ')
for i in range(M.nO):
    print('   > i     = ', i)
    print('   > Nodes = ', M.objects[i].nodes)
    print('   > Parts = ', M.objects[i].parts)
print('')


print(' > Testing preconditioner:')
P = bddc.bddc(n,A,M)
for i in range(P.nP):
    print('   > i      = ', i)
    print('   > nodesI = ', P.fine[i].nodesI)
    print('   > nodesB = ', P.fine[i].nodesB)
    print('   > constr = ', P.fine[i].constr)
    # print('   > Aii    = \n', P.fine[i].Aii.todense())
    # print('   > Aib    = \n', P.fine[i].Aib.todense())
    # print('   > Abb    = \n', P.fine[i].Abb.todense())
    # print('   > C      = \n', P.fine[i].C.todense())
    # print('   > Wc     = ', P.fine[i].Wc.transpose())
    # print('   > Wb     = ', P.fine[i].Wb.transpose())

print('')

# Solve
print(' > Solving Problem: ')
solver = pcg.pcg(n,A,b,M)
solver.solve(x0,tol=1.e-10)
print('')


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1,1,1)
# x = np.linspace(0,1,nSide)
# y = np.reshape(solver.sol,(nSide,nSide))
# ax1.contourf(y, levels=100)

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(1,1,1)
# ax2.spy(A)

print('Converged = ', solver.converged)
print('Num Iter  = ', solver.numIter)
#print('Solution  = ', solver.sol.transpose())
print('Residual  = ', np.linalg.norm(np.abs(A@solver.sol-b).transpose()))



plt.show()

