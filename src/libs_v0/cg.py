
###############################################################################
#####                     CONJUGATE GRADIENT CLASS                        #####
###############################################################################

import numpy as np
import math
from scipy.sparse import csr_matrix as csr


class cg ():
    def __init__(self, n, A, b, mesh):
        self.n = n # Problem size
        self.A = A # System matrix in CSR format (nxn)
        self.b = b # System vector

        self.mesh = mesh

        self.sol = None
        return 


    def solve(self,x0,tol=1.e-10):
        # Initialisation
        k = 0
        xk = np.array(x0)
        rk = self.b - self.A @ xk
        dk = rk

        # Main loop
        while (np.linalg.norm(rk) > tol and k < 2.0*self.n): 
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
