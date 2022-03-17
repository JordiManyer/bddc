
###############################################################################
#####               PRECONDITIONED CONJUGATE GRADIENT CLASS               #####
###############################################################################

import numpy as np
import math
from scipy.sparse import csr_matrix as csr
from bddc import bddc
# from mpi4py import MPI


class pcg ():
    def __init__(self, n, A, b, mesh):
        self.n = n # Problem size
        self.A = A # System matrix in CSR format (nxn)
        self.b = b # System vector

        self.mesh = mesh
        self.BDDC = bddc(n,A,mesh)

        self.converged = False
        self.numIter   = 0
        self.sol       = None
        return 


    def solve(self,x0,tol=1.e-10):
        # Initialisation
        k = 0
        xk = np.array(x0)
        rk = self.b - self.A @ xk
        zk = self.BDDC.applyBDDC(np.array(rk))
        dk = zk

        # Main loop
        while (np.linalg.norm(rk) > tol and k < 10.0*self.n): 
            Adk = self.A @ dk

            # Alpha_k and x_{k+1}
            alpha = (zk.transpose() @ rk) / (dk.transpose() @ Adk)
            xk = xk + alpha * dk

            # r_{k+1}, we save r_k
            rkm1 = rk
            rk = rk - alpha * Adk

            # z_{k+1}, we save z_k
            zkm1 = zk
            zk = self.BDDC.applyBDDC(np.array(rk))

            # Beta_k and d_{k+1}
            beta = (zk.transpose() @ rk) / (zkm1.transpose() @ rkm1)
            dk = zk + beta * dk

            k = k+1
        self.sol = xk
        self.numIter = k
        self.converged = (np.linalg.norm(rk) <= tol)
