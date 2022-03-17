import numpy as np 
import math
from scipy.sparse import csr_matrix as csr
import scipy.sparse.linalg


class coarseProc():
    def __init__(self):
        self.nP = 0          # Number of fine processors

        self.nO = 0          # Total number of objects
        self.com_size = None # Number of objects in each processor, size(nP+1)
        self.com_obj  = None # Objects in each processor, size(nO)
        self.W = None        # Weights of each object

        self.nC = 0          # Total number of constraints
        self.cst_size = None # Number of constraints in each processor
        self.cst_cst  = None # Constraints in each processor
        self.S0 = None       # Coarse system matrix
        self.invS0 = None


    def createGlobalOrdering(self,nP,fineProcs):
        self.nP = nP

        # Get number of local objects
        self.com_size = np.zeros(nP+1,dtype=int)
        for iP in range(nP): self.com_size[iP+1] = fineProcs[iP].nO    # <>GlobalComunication<> MPI_Gather
        for iP in range(nP): self.com_size[iP+1] += self.com_size[iP]

        # Get local object ids and number of constraints per object
        #  > id1    -> Object edge with minimum global numbering
        #  > id2    -> Processor containing this object with minimum rank
        #  > numCst -> Number of constraints for each of the objects
        id1    = np.zeros(self.com_size[nP],dtype=int)
        id2    = np.zeros(self.com_size[nP],dtype=int)
        numCst = np.zeros(self.com_size[nP],dtype=int)
        for iP in range(nP):
            id1[self.com_size[iP]:self.com_size[iP+1]]    = fineProcs[iP].obj_id1[:fineProcs[iP].nO] # <>GlobalComunication<> MPI_Gather
            id2[self.com_size[iP]:self.com_size[iP+1]]    = fineProcs[iP].obj_id2[:fineProcs[iP].nO] # <>GlobalComunication<> MPI_Gather
            numCst[self.com_size[iP]:self.com_size[iP+1]] = fineProcs[iP].obj_cts                    # <>GlobalComunication<> MPI_Gather

        # Create global ordering for objects and constraints
        self.nO = 0
        self.nC = 0
        self.com_obj = np.zeros(self.com_size[nP],dtype=int)
        self.cst_size = np.zeros(self.nP+1,dtype=int)
        self.cst_cst  = np.zeros(np.sum(numCst),dtype=int)
        cst_starts    = np.zeros(self.com_size[nP],dtype=int)
        for iP in range(nP):
            self.cst_size[iP+1] = self.cst_size[iP]
            for i in range(self.com_size[iP],self.com_size[iP+1]):
                if (iP == id2[i]): # Cannot have been numbered yet -> New object
                    # Object
                    self.com_obj[i] = self.nO
                    self.nO += 1
                    # Constraints
                    cst_starts[i] = self.cst_size[iP+1]
                    for k in range(numCst[i]):
                        self.cst_cst[self.cst_size[iP+1]] = self.nC
                        self.cst_size[iP+1] += 1
                        self.nC += 1

                else:              # This object has already been numbered -> Find its numbering
                    for j in range(self.com_size[id2[i]],self.com_size[id2[i]+1]):
                        if (id1[j] == id1[i]) : 
                            # Object
                            self.com_obj[i] = self.com_obj[j]
                            # Constraints
                            for k in range(numCst[i]):
                                self.cst_cst[self.cst_size[iP+1]] = self.cst_cst[cst_starts[j]+k]
                                self.cst_size[iP+1] += 1

        # Get weights for the constraints
        self.W = np.zeros(self.nC)
        for iC in range(self.cst_size[self.nP]): # Weighting with partition count
            self.W[self.cst_cst[iC]] += 1 
        self.W = 1.0/self.W


    def initCoarse(self,nP,fineProcs):
        # Assemble coarse problem
        # TODO: This should be done directly in CSR format, using a connectivity graph 
        #       to find the fill-in structure.

        # Assemble coarse system
        S0 = np.zeros((self.nC,self.nC))
        for i in range(nP):
            S0i = - fineProcs[i].Phi.transpose() @ fineProcs[i].C_csr.transpose() @ fineProcs[i].Lambda 
            # S0i_bis = fineProcs[i].Phi.transpose() @ fineProcs[i].A @ fineProcs[i].Phi
            # print('>>> Coarse:: ||S0i - S0i_bis|| = ', np.linalg.norm(S0i - S0i_bis)/np.linalg.norm(S0i))

            const = self.cst_cst[self.cst_size[i]:self.cst_size[i+1]]
            S0[np.ix_(const,const)] += S0i # <>GlobalComunication<> MPI_Gather
        self.S0 = csr(S0)

        # Factorize coarse system
        self.invS0 = scipy.sparse.linalg.factorized(self.S0) 

