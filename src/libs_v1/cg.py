
###############################################################################
#####                      CONJUGATE GRADIENT CLASS                       #####
###############################################################################

import numpy as np
import math
from scipy.sparse import csr_matrix as csr
from bddc import *


class cg ():
    def __init__(self, fineProcs):
        self.nP = len(fineProcs)
        self.fine = fineProcs

        print('  > INITIALISING BDDC')
        self.P = bddc(fineProcs) # Will be unused, needed to setup self.fine[iP].W
        self.P.init()            
        self.getGlobalOrdering()

        self.n = 0     # Problem size
        for iP in range(self.nP):
            self.n = max(self.n,np.amax(self.fine[iP].globalTag)+1)

        self.b = np.zeros((self.n,1))                      # System vector
        for iP in range(self.nP):
            self.b[self.fine[iP].globalTag] += self.fine[iP].W * self.fine[iP].b
        self.b = self.b.reshape((self.n,1))

        self.converged = False
        self.numIter   = 0
        self.sol       = None
        self.residual  = -1
        return 


    def solve(self,x0,tol=1.e-10):
        print('  > STARTING CONVERGENCE LOOP')
        # Initialisation
        k = 0
        normB = np.linalg.norm(self.b)
        xk = np.array(x0)
        rk = self.b - self.multiplyA(xk)
        dk = np.array(rk)

        # Main loop
        while (np.linalg.norm(rk) > tol * normB and k < 100.0*self.n): 
            if (k % 100 == 0) : 
                print('    > Iteration k = ', k, ' , Residual = ', np.linalg.norm(rk))
                print('        > ||dk|| = ', np.linalg.norm(dk))
                if (k > 0): 
                    print('        > alpha  = ', alpha)
                    print('        > beta   = ', beta)
            Adk = self.multiplyA(dk)

            # Alpha_k and x_{k+1}
            alpha = (dk.transpose() @ rk) / (dk.transpose() @ Adk)
            xk = xk + alpha * dk

            # r_{k+1}, we save r_k
            rkm1 = rk
            rk = rk - alpha * Adk

            # Beta_k and d_{k+1}
            beta = (rk.transpose() @ rk) / (rkm1.transpose() @ rkm1)
            dk = rk + beta * dk

            # if (k % 1000 == 0) : print('      > alpha = ' , alpha, ', beta = ', beta)
            # if (k % 1000 == 0) : print('      > 1 = ' , rk.transpose() @ rk, ', 2 = ', rkm1.transpose() @ rkm1)

            k = k+1
        self.sol = xk
        self.numIter = k
        self.residual  = np.linalg.norm(rk)
        self.converged = (self.residual <= tol)


    def multiplyA(self,x):
        y = np.zeros((self.n,1))
        for iP in range(self.nP):
            nodes = self.fine[iP].globalTag
            y[nodes] += self.fine[iP].A @ x[nodes]
        return y


    def getGlobalOrdering(self):
        # Initialise
        for iP in range(self.nP):
            self.fine[iP].globalTag = np.zeros(self.fine[iP].n,dtype=int)

        counter = 0

        # Interior edges
        for iP in range(self.nP):
            nI = self.fine[iP].nI
            self.fine[iP].globalTag[:nI] = np.arange(counter,counter+nI)
            counter += nI

        # Boundary edges
        d = {}
        for iP in range(self.nP):
            nI = self.fine[iP].nI
            nB = self.fine[iP].nB
            for iB in range(nI,nI+nB):
                iB_com  = np.where(self.fine[iP].com_loc == iB)[0][0]
                iB_glob = self.fine[iP].com_glob[iB_com]

                if (iB_glob in d.keys()) :
                    self.fine[iP].globalTag[iB] = d[iB_glob]
                else :
                    self.fine[iP].globalTag[iB] = counter 
                    d[iB_glob] = counter
                    counter += 1

