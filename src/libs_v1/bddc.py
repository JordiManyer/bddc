import numpy as np 
import math
import scipy.sparse.linalg
from scipy.sparse import csr_matrix as csr
from scipy.sparse import bmat

from fineProc import *
from coarseProc import *


class bddc():
    def __init__(self, fineProcs):
        self.nP    = len(fineProcs)
        self.fine  = fineProcs
        self.coarse = coarseProc()


    def init(self):
        for iP in range(self.nP):                           # Done in parallel (local)
            self.fine[iP].initAll()

        self.coarse.createGlobalOrdering(self.nP,self.fine) # <>GlobalComunication<>
        self.coarse.initCoarse(self.nP,self.fine)           # <>GlobalComunication<>


    def apply(self, r):
        n = r.size

        self.n = n
        for i in range(self.nP):
            # Although saved in fine, this info does not need to leave the coarse proc (therefore no global comm needed)
            self.fine[i].constr = self.coarse.cst_cst[self.coarse.cst_size[i]:self.coarse.cst_size[i+1]]

        # 1) First interior correction
        u0 = np.zeros((n,1))
        r1 = np.array(r)
        for i in range(self.nP):
            u0[self.fine[i].nodesI]  = self.fine[i].invAii(r1[self.fine[i].nodesI])
            r1[self.fine[i].nodesI]  = 0.0
            r1[self.fine[i].nodesB] -= self.fine[i].Aib.transpose() @ u0[self.fine[i].nodesI] # <Local communication> Share r1 between NN


        # 2) BDDC correction: A, B can be done in parallel in fine, coarse procs 
        u1 = np.zeros((n,1))
        # 2.A) Fine Correction 
        for i in range(self.nP): 
            rF = np.zeros((self.fine[i].nI + self.fine[i].nB + self.fine[i].nC,1))
            rF[:self.fine[i].nI+self.fine[i].nB] = self.fine[i].W * r1[self.fine[i].globalTag]
            zF = self.fine[i].invFine(rF)

            u1[self.fine[i].globalTag] += self.fine[i].W * zF[:self.fine[i].nI+self.fine[i].nB]  # <Local communication> Share u1 between NN

        # 2.B) Coarse Correction
        rC = np.zeros((self.coarse.nC,1))
        for i in range(self.nP):
            rCi = self.fine[i].Phi.transpose() @ (self.fine[i].W * r1[self.fine[i].globalTag])
            rC[self.fine[i].constr] += rCi # <Global communication> MPI_Gather r0i into coarse proc

        zC = self.coarse.invS0(rC)

        for i in range(self.nP):
            zCi = self.fine[i].Phi @ zC[self.fine[i].constr] # <Global communication> MPI_Scatter r0 into fine procs
            u1[self.fine[i].globalTag] += self.fine[i].W * zCi  # <Local communication> Share u1 between NN

        # 3) Second interior correction
        r2 = np.array(r1 - self.multiplyA(u1)) # <Local communication> Share r2 between NN
        u2 = np.zeros((n,1))
        for i in range(self.nP):
            u2[self.fine[i].nodesI] = self.fine[i].invAii(r2[self.fine[i].nodesI])
        
        z = u0 + u1 + u2
        return z


    def multiplyA(self,x):
        y = np.zeros((self.n,1))
        for iP in range(self.nP):
            nodes = self.fine[iP].globalTag
            y[nodes] += self.fine[iP].A @ x[nodes]
        return y