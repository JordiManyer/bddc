###############################################################################
#####                      CONJUGATE GRADIENT CLASS                       #####
###############################################################################

import numpy as np
import math
from scipy.sparse import csr_matrix as csr


class cg_serial ():
    def __init__(self, A , b):

        self.n = b.size
        self.b = b               
        self.A = A

        self.converged = False
        self.numIter   = 0
        self.sol       = None
        self.residual  = -1
        return 


    def solve(self,x0,tol=1.e-10):
        print('  > STARTING CONVERGENCE LOOP')
        # Initialisation
        k = 0
        xk = np.array(x0)
        rk = self.b - self.A @ xk
        dk = rk

        # Main loop
        while (np.linalg.norm(rk) > tol and k < 100.0*self.n): 
            if (k % 1000 == 0) : print('    > Iteration k = ', k, ' , Residual = ', np.linalg.norm(rk))
            Adk = self.A @ dk

            # Alpha_k and x_{k+1}
            alpha = (dk.transpose() @ rk) / (dk.transpose() @ Adk)
            xk = xk + alpha * dk

            # r_{k+1}, we save r_k
            rkm1 = rk
            rk = rk - alpha * Adk

            # Beta_k and d_{k+1}
            beta = (rk.transpose() @ rk) / (rkm1.transpose() @ rkm1)
            dk = rk + beta * dk

            k = k+1
        self.sol = xk
        self.numIter = k
        self.residual  = np.linalg.norm(rk)
        self.converged = (self.residual <= tol)

