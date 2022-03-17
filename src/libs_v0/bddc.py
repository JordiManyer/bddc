import numpy as np 
import math
import scipy.sparse.linalg
from scipy.sparse import csr_matrix as csr
from scipy.sparse import bmat

# Class containing all the data that should be distributed per proc
class fineProc():
    def __init__(self):
        self.nI      = 0     # Number of interior nodes
        self.nB      = 0     # Number of interface nodes
        self.nC      = 0     # Number of local constraints

        self.nodesI  = []    # List of interor nodes
        self.nodesB  = []    # List of interface nodes
        self.constr  = []    # List of constraints

        self.Aii     = None  # Interior-Interior matrix
        self.Aib     = None  # Interior-Interface matrix
        self.Abb     = None  # Interface-Interface matrix
        self.C       = None  # Constraints matrix
        self.Wc      = None  # Weights for constraints
        self.Wb      = None  # Weights for interface nodes

        self.Phi     = None  # Local coarse eigenvectors
        self.Lambda  = None  # Local coarse lagrange multipliers

        self.invAii  = None
        self.invFine = None

class coarseProc():
    def __init__(self):
        self.nC = 0          # Total number of constraints
        self.S0 = None       # Coarse system matrix



class bddc():
    def __init__(self, n, A, mesh):
        self.n  = n 

        self.nP = mesh.nP
        self.fine = []
        for i in range(self.nP):
            self.fine.append(fineProc())
        self.initFine(n, A, mesh)

        self.coarse = coarseProc()
        self.initCoarse(n, A, mesh)


    def initFine(self, n, A, mesh):
        # Collect local indexes
        for k in range(n):
            if (mesh.parts[k] == -2 ): # Interface node - edge
                for j in range(len(mesh.nodeMap[k])):
                    iP = mesh.parts[mesh.nodeMap[k][j]]
                    if ( iP >= 0 and k not in self.fine[iP].nodesB):
                        self.fine[iP].nB += 1
                        self.fine[iP].nodesB.append(k)
            elif (mesh.parts[k] == -3 ): # Interface node - corner
                for j in mesh.nodeMap[k]:
                    for i in mesh.nodeMap[j]:
                        iP = mesh.parts[i]
                        if ( iP >= 0 and k not in self.fine[iP].nodesB):
                            self.fine[iP].nB += 1
                            self.fine[iP].nodesB.append(k)
            else:                      # Interior node
                iP = mesh.parts[k]
                self.fine[iP].nI += 1
                self.fine[iP].nodesI.append(k)

        # Collect local matrices 
        for i in range(self.nP):
            self.fine[i].Aii = A[np.ix_(self.fine[i].nodesI,self.fine[i].nodesI)]
            self.fine[i].Aib = A[np.ix_(self.fine[i].nodesI,self.fine[i].nodesB)]
            self.fine[i].Abb = A[np.ix_(self.fine[i].nodesB,self.fine[i].nodesB)]

        # Manage constraints: Select local objects and create C
        for i in range(mesh.nO):
            for j in range(len(mesh.objects[i].parts)): 
                iP = mesh.objects[i].parts[j]
                self.fine[iP].nC += 1
                self.fine[iP].constr.append(i)

        for i in range(self.nP):
            rows = np.zeros(self.fine[i].nC+1,dtype=int)
            cols = np.zeros(self.fine[i].nB,dtype=int)
            data = np.zeros(self.fine[i].nB)

            for j in range(self.fine[i].nC):
                iC = self.fine[i].constr[j]
                nNodes = len(mesh.objects[iC].nodes)
                rows[j+1] = rows[j]
                for k in range(nNodes):
                    local_index     = self.fine[i].nodesB.index(mesh.objects[iC].nodes[k])
                    cols[rows[j+1]] = local_index
                    data[rows[j+1]] = 1/nNodes # Mean along the object
                    rows[j+1] += 1
            self.fine[i].C = csr((data,cols,rows), shape=(self.fine[i].nC,self.fine[i].nB))

        # Invert local problems
        for i in range(self.nP):
            self.fine[i].invAii  = scipy.sparse.linalg.factorized(self.fine[i].Aii)

            Aaux = bmat([[self.fine[i].Aii              ,  self.fine[i].Aib   ,  None                       ] ,
                         [self.fine[i].Aib.transpose()  ,  self.fine[i].Abb   ,  self.fine[i].C.transpose() ] ,
                         [None                          ,  self.fine[i].C     ,  None                       ] ])
            self.fine[i].invFine = scipy.sparse.linalg.factorized(Aaux)
            
        return


    def initCoarse(self, n, A, mesh):
        # Get weights for interface nodes
        for i in range(self.nP):
            self.fine[i].Wc = np.zeros((self.fine[i].nC,1))
            self.fine[i].Wb = np.zeros((self.fine[i].nB,1))
            for j in range(self.fine[i].nC): # Weighting with partition count
                self.fine[i].Wc[j] = 1.0/len(mesh.objects[self.fine[i].constr[j]].parts)
                for k in mesh.objects[self.fine[i].constr[j]].nodes:
                    self.fine[i].Wb[self.fine[i].nodesB.index(k)] = self.fine[i].Wc[j]
        
        # Get local eigenvectors from Newmann problem
        for i in range(self.nP):
            self.fine[i].Phi    = np.zeros((self.fine[i].nB, self.fine[i].nC))
            self.fine[i].Lambda = np.zeros((self.fine[i].nC, self.fine[i].nC))
            for j in range(self.fine[i].nC):
                x = np.zeros(self.fine[i].nI + self.fine[i].nB + self.fine[i].nC)
                x[self.fine[i].nI + self.fine[i].nB + j] = 1.0
                y = self.fine[i].invFine(x)

                self.fine[i].Phi[:,j]    = y[self.fine[i].nI:self.fine[i].nI+self.fine[i].nB]
                self.fine[i].Lambda[:,j] = y[self.fine[i].nI+self.fine[i].nB:]

        # Assemble coarse problem
        # TODO: This should be done directly in CSR format, using a connectivity graph 
        #       to find the fill-in structure.
        self.coarse.nC = mesh.nO
        S0 = np.zeros((self.coarse.nC,self.coarse.nC))
        for i in range(self.nP):
            S0i = -self.fine[i].Phi.transpose()@self.fine[i].C.transpose()@self.fine[i].Lambda
            for j in range(self.fine[i].nC): # Weighting with partition count
                S0i[j,:] *= self.fine[i].Wc[j]
                S0i[:,j] *= self.fine[i].Wc[j]

            S0[np.ix_(self.fine[i].constr,self.fine[i].constr)] += S0i
        self.coarse.S0 = csr(S0)

        # Factorize coarse system
        self.coarse.invS0 = scipy.sparse.linalg.factorized(self.coarse.S0)

        return


    def interiorCorrection(self, r):
        z = np.zeros((self.n,1))
        for i in range(self.nP):
            z[self.fine[i].nodesI] = self.fine[i].invAii(r[self.fine[i].nodesI])
        return z


    def applyBDDC(self, r):
        z = np.zeros((self.n,1))

        # First interior correction
        for i in range(self.nP):
            z[self.fine[i].nodesI] = self.fine[i].invAii(r[self.fine[i].nodesI])
            r[self.fine[i].nodesB] -= self.fine[i].Aib.transpose() @  z[self.fine[i].nodesI]

        # Fine Correction
        for i in range(self.nP): 
            x = np.zeros((self.fine[i].nI + self.fine[i].nB + self.fine[i].nC,1))
            x[self.fine[i].nI:self.fine[i].nI+self.fine[i].nB] = r[self.fine[i].nodesB]
            y = self.fine[i].invFine(x)

            z[self.fine[i].nodesB] += self.fine[i].Wb * y[self.fine[i].nI:self.fine[i].nI+self.fine[i].nB]

        # Coarse Correction
        r0 = np.zeros((self.coarse.nC,1))
        for i in range(self.nP):
            r0i = self.fine[i].Phi.transpose()@r[self.fine[i].nodesB]
            r0[self.fine[i].constr] += self.fine[i].Wc * r0i

        z0 = self.coarse.invS0(r0)

        for i in range(self.nP):
            z0i = self.fine[i].Phi @ z0[self.fine[i].constr]
            z[self.fine[i].nodesB] += self.fine[i].Wb * z0i

        # Second interior correction
        for i in range(self.nP):
            aux = self.fine[i].Aib @  z[self.fine[i].nodesB]
            z[self.fine[i].nodesI] -= self.fine[i].invAii(aux)

        return z.reshape((z.size,1))