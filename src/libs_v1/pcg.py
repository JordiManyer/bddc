
###############################################################################
#####               PRECONDITIONED CONJUGATE GRADIENT CLASS               #####
###############################################################################

import numpy as np
import math
from scipy.sparse import csr_matrix as csr
from bddc import *
from jacobi import *


class pcg ():
    def __init__(self, fineProcs):
        self.nP = len(fineProcs)
        self.fine = fineProcs

        self.P = bddc(fineProcs)
        #self.P = jacobi(fineProcs)
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
        print('    > normB = ', normB)
        xk = np.array(x0)
        rk = self.b - self.multiplyA(xk)
        zk = self.P.apply(np.array(rk))
        dk = np.array(zk)

        # Main loop
        while (np.linalg.norm(rk) > tol * normB and k < 100.0*self.n): 
            if (k % 10 == 0) : 
                print('    > Iteration k = ', k, ' , Residual = ', np.linalg.norm(rk))
                print('        > ||dk|| = ', np.linalg.norm(dk))
                print('        > ||zk|| = ', np.linalg.norm(zk))
                if (k > 0): 
                    print('        > ||Adk|| = ', np.linalg.norm(Adk))
                    print('        > alpha  = ', alpha)
                    print('        > beta   = ', beta)
                    print('        > zk·rk   = ', zk.transpose() @ rk)
                    print('        > dk·rk   = ', dk.transpose() @ rk)
                    aux1 = 0 
                    aux2 = 0 
                    for iP in range(self.nP):
                        aux1 += np.dot(rk[self.fine[iP].nodesI].flatten(),zk[self.fine[iP].nodesI].flatten())
                        aux2 += 0.5*np.dot(rk[self.fine[iP].nodesB].flatten(),zk[self.fine[iP].nodesB].flatten())

                    print('        > zI·rI = ', aux1)
                    print('        > zB·rB = ', aux2)


            Adk = self.multiplyA(dk)

            # Alpha_k and x_{k+1}
            alpha = (zk.transpose() @ rk) / (dk.transpose() @ Adk)
            xk = xk + alpha * dk

            # r_{k+1}, we save r_k
            rkm1 = np.array(rk)
            rk = rk - alpha * Adk

            # z_{k+1}, we save z_k
            zkm1 = np.array(zk)
            zk = self.P.apply(np.array(rk))

            # Beta_k and d_{k+1}
            beta = (zk.transpose() @ rk) / (zkm1.transpose() @ rkm1) # Fletcher-Reeves formula (Fixed PCG)
            # beta = (rk.transpose() @ (zk - zkm1)) / (rkm1.transpose() @ zkm1)   # Polak-Ribière formula (Flexible PCG)
            dk = zk + beta * dk

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

        # Get global tags for interior/boundary dofs only 
        for iP in range(self.nP):
            self.fine[iP].nodesI = self.fine[iP].globalTag[:self.fine[iP].nI]
            self.fine[iP].nodesB = self.fine[iP].globalTag[self.fine[iP].nI:]

