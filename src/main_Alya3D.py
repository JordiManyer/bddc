###############################################################################
#####                             SCRIPT TO COMPARE SOLVERS               #####
###############################################################################
import sys
sys.path.append("./libs_v1")

import numpy as np 
from scipy.sparse import csr_matrix as csr
from scipy.sparse.linalg import spsolve as spsolve
import matplotlib.pyplot as plt

import alyaIO as IO
from fineProc import *
from coarseProc import *
from bddc import *
from pcg import *
from cg import *
from cg_serial import *

# --------------------------------------------------------------------------
# Script parameters
inputDir  = 'DOF166796'

inputPath = '../input/' + inputDir
caseName  = ''
nP = 0
if (inputDir == 'DOF166796' ):
    caseName = 'TS-SLAB-HEX08-BULK'
    nP = 15
elif (inputDir == 'DOF669780' ):
    caseName = 'TS-SLAB-HEX08-BULK'
    nP = 23

# --------------------------------------------------------------------------
# Read data from Alya
fine = []
print('> READING INPUTS')
for i in range(nP):
    fine.append(fineProc())
    IO.readFineProc(fine[i], inputDir, caseName, i)

# Scaling 
if (False):
    solver_aux = pcg(fine)
    bnorm = np.linalg.norm(solver_aux.b)
    print('>> bnorm = ', bnorm)
    for iP in range(nP):
        fine[iP].A /= bnorm
        fine[iP].b /= bnorm


# --------------------------------------------------------------------------
print('> PCG SOLVER')
solver_pcg = pcg(fine)
x0 = 2*np.ones((solver_pcg.n,1))
solver_pcg.solve(x0,tol=1.e-8)

print('Converged = ', solver_pcg.converged)
print('Num Iter  = ', solver_pcg.numIter)
print('Solution  = ', solver_pcg.sol.transpose())
print('Residual  = ', solver_pcg.residual)

# Assemble global matrix and vector
b_glob = solver_pcg.b
A_glob = csr((solver_pcg.n,solver_pcg.n))
for iP in range(nP):
    nodes = solver_pcg.fine[iP].globalTag
    A_glob[np.ix_(nodes,nodes)] += solver_pcg.fine[iP].A

print('Residual2 = ', np.linalg.norm(b_glob - A_glob @ solver_pcg.sol))
