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
inputDir  = 'DOF33062'

inputPath = '../input/' + inputDir
caseName  = ''
nP = 0
if (inputDir == 'DOF40' or inputDir == 'DOF3564' ):
    caseName = 'wire-TRI03'
    nP = 4
elif (inputDir == 'DOF33062'):
    caseName = 'TS-TAPE-MIXED-ANG'
    nP = 8

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
    for iP in range(nP):
        fine[iP].A /= bnorm
        fine[iP].b /= bnorm


# --------------------------------------------------------------------------
print('> PCG SOLVER')
solver_pcg = pcg(fine)
x0 = 2*np.ones((solver_pcg.n,1))
solver_pcg.solve(x0,tol=1.e-16)

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


# --------------------------------------------------------------------------
print('> CG SOLVER')
solver_cg = cg(fine)
x0 = 2*np.ones((solver_cg.n,1))
solver_cg.solve(x0,tol=1.e-1)
print('Converged = ', solver_cg.converged)
print('Num Iter  = ', solver_cg.numIter)
print('Solution  = ', solver_cg.sol.transpose())
print('Residual  = ', solver_cg.residual)
print('Residual2 = ', np.linalg.norm(A_glob @ solver_cg.sol - b_glob))


# --------------------------------------------------------------------------
print('> EXACT SOLVER')
sol_exact   = spsolve(A_glob,b_glob).reshape((solver_cg.n,1))

print('Solution  = ', sol_exact.transpose())
print('Residual  = ', np.linalg.norm(A_glob@sol_exact-b_glob))


# --------------------------------------------------------------------------
print('> Comparing Solutions: ')
print('  > PCG vs CG')
print('      ||sol_pcg - sol_cg||_inf = ', np.linalg.norm(solver_pcg.sol - solver_cg.sol,ord=np.inf))
print('      ||sol_pcg - sol_cg||_1   = ', np.linalg.norm(solver_pcg.sol - solver_cg.sol,ord=1))
print('      ||sol_pcg - sol_cg||_2   = ', np.linalg.norm(solver_pcg.sol - solver_cg.sol,ord=2))
print('  > PCG vs EXACT')
print('      ||sol_pcg - sol_ex||_inf = ', np.linalg.norm(solver_pcg.sol - sol_exact,ord=np.inf))
print('      ||sol_pcg - sol_ex||_1   = ', np.linalg.norm(solver_pcg.sol - sol_exact,ord=1))
print('      ||sol_pcg - sol_ex||_2   = ', np.linalg.norm(solver_pcg.sol - sol_exact,ord=2))
print('  > CG vs EXACT')
print('      ||sol_cg  - sol_ex||_inf = ', np.linalg.norm(sol_exact - solver_cg.sol,ord=np.inf))
print('      ||sol_cg  - sol_ex||_1   = ', np.linalg.norm(sol_exact - solver_cg.sol,ord=1))
print('      ||sol_cg  - sol_ex||_2   = ', np.linalg.norm(sol_exact - solver_cg.sol,ord=2))




doPlots = False
if (doPlots):
    X = np.arange(0,solver_cg.n)
    diff_PCG_EXACT    = np.abs(solver_pcg.sol - sol_exact)
    diff_PCG_CG = np.abs(solver_pcg.sol - solver_cg.sol)
    diff_EXACT_CG  = np.abs(solver_cg.sol - sol_exact)

    fig1,ax1 = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=(13, 5))
    ax1[0].plot(X,diff_PCG_CG)
    ax1[1].plot(X,diff_PCG_EXACT)
    ax1[2].plot(X,diff_EXACT_CG)
    ax1[0].set_title('PCG - CG')
    ax1[1].set_title('PCG - EXACT')
    ax1[2].set_title('CG  - EXACT')

    for iP in range(nP):
        nodesI = fine[iP].globalTag[:fine[iP].nI]
        nodesB = fine[iP].globalTag[fine[iP].nI:]
        ax1[3].plot(nodesI,iP*np.ones(nodesI.size),'.')
        ax1[3].plot(nodesB,nP*np.ones(nodesB.size),'.k')


    fig2,ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=(13, 5))
    ax2[0].plot(X,sol_exact)
    ax2[1].plot(X,solver_pcg.sol)
    ax2[2].plot(X,solver_cg.sol)
    ax2[0].set_title('EXACT')
    ax2[1].set_title('PCG')
    ax2[2].set_title('CG ')

    for iP in range(nP):
        nodesI = fine[iP].globalTag[:fine[iP].nI]
        nodesB = fine[iP].globalTag[fine[iP].nI:]
        ax2[3].plot(nodesI,iP*np.ones(nodesI.size),'.')
        ax2[3].plot(nodesB,nP*np.ones(nodesB.size),'.k')

    plt.show()
