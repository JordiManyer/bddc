import numpy as np 
import math
import scipy.sparse.linalg
from scipy.sparse import csr_matrix as csr
from scipy.sparse import bmat




# Class containing all the data that should be distributed per proc
class fineProc():
    def __init__(self):
        # Local dofs and data 
        self.dim        = 2
        self.n          = 0    # Local number of dofs
        self.nI         = 0    # Number of interior nodes
        self.nB         = 0    # Number of interface nodes

        self.A          = None # Local problem matrix
        self.b          = None # Local problem vector

        # Mesh info
        self.nE         = 0    # Number of elements
        self.nN         = 0    # Number of nodes
        self.nNbors     = 0    # Number of edges per element
        self.elemEdge   = None # Element-edge connectivity, size(nE,nNbors)
        self.signs      = None # Sign map for each edge, size(nE,nNbors)
        self.eemap_size = None # CSR Indexes for the edge-edge map, size(nE+1)
        self.eemap      = None # CSR data for the edge-edge map, size(eemap_size[nE])
        self.edgeNodes  = None # Nodes for each edge, size(n,2)
        self.nemap_size = None # CSR Indexes for the node-edge map
        self.nemap      = None # CSR data for the node-edge map

        # Communication 
        self.rank      = 0     # Rank of this processor
        self.nP        = 0     # Number of neighboring procs
        self.procs     = None  # List of neighboring procs
        self.com_size  = None  # Communication vector sizes for dofs
        self.com_loc   = None  # Communication vector local indexes for dofs
        self.com_glob  = None  # Communication vector global indexes for dofs
        self.globalTag = None  # Global ordering of each edge

        # BDDC-specific variables
        self.nO        = 0     # Number of local objects
        self.obj_size  = None  # Number of edges for each object
        self.obj_dofs  = None  # Edges for each object, written in obj_size[iO]:obj_size[iO+1]
        self.obj_node  = None  # Interior nodes for each object, written in obj_size[iO]:obj_size[iO+1]-1
        self.obj_id1   = None  # Object ID1: Minimal Global ID of object edges
        self.obj_id2   = None  # Object ID2: Minimal global ID of object neighboring partitions
        self.obj_sign  = None  # Orientation of edges inside the object
        self.obj_cts   = None  # Number of constraints for each object

        self.nC        = 0     # Number of local constraints
        self.C_size    = None  # Sizes for C
        self.C         = None  # Local Constraint matrix
        self.W         = None  # Weights for all edges

        self.Phi       = None  # Local coarse eigenvectors
        self.Lambda    = None  # Local coarse lagrange multipliers

        self.Aib       = None
        self.invAii    = None
        self.invFine   = None


    def initAll(self):
        self.getObjects()
        self.getConstraints()
        self.initFine()


    def getObjects(self):
        # Get neighboring processors for every boundary edge
        numNbors = np.zeros(self.nB,dtype=int)
        nbors    = -np.ones((self.nB*self.nP),dtype=int)
        for iP in range(self.nP):
            for iE in range(self.com_size[iP],self.com_size[iP+1]):
                e = self.com_loc[iE]-self.nI
                nbors[e*self.nP + numNbors[e]] = self.procs[iP]
                numNbors[e] += 1

        # Select objects: 
        #    > In 2D, we only have edges and all edges border 2 processors (i.e. numNbors[e] == 1)
        #    > In 3D, Faces border 2 processors (i.e. numNbors[e] == 1) 
        #      and Edges border > 2 processors  (i.e. numNbors[e] > 1 )
        maxNumObjects = int(self.nP*(self.nP+1)/2) # nP choose 2
        self.obj_size = np.zeros(maxNumObjects+1,dtype=int)
        self.obj_dofs = np.zeros(self.nB,dtype=int)
        self.obj_sign = np.zeros(self.nB,dtype=int)
        self.obj_node = -np.ones((self.nB,2),dtype=int)
        self.obj_id1  = np.zeros(maxNumObjects,dtype=int)
        self.obj_id2  = np.zeros(maxNumObjects,dtype=int)

        iO = 0
        visited = np.zeros(self.nB, dtype=bool)
        if (self.dim == 3) : visited = numNbors+1 < 3 # Eliminate face edges (set them as True)
        for iB in range(self.nB):
            if (not visited[iB]): # Start a new object
                self.obj_dofs[self.obj_size[iO+1]] = iB + self.nI
                self.obj_sign[self.obj_size[iO+1]] = 1

                i_com = np.where(self.com_loc == iB + self.nI)[0][0]
                self.obj_id1[iO] = self.com_glob[i_com]
                self.obj_id2[iO] = int(min(self.rank,np.amin(nbors[iB*self.nP:iB*self.nP+numNbors[iB]])))
                index_id1 = self.obj_size[iO+1]
                self.obj_size[iO+1] += 1

                iNode  = 0 # Restart node numbering

                # Follow the edge chain until exhausted, 
                # only a 2-position queue is needed (2 outmost edges of the chain)
                q      = [iB + self.nI,0]
                q_sign = [1,0]
                iq     = 1 
                while(iq > 0):
                    iq = iq - 1
                    e = q[iq]            # Current edge (local numbering)
                    ie = e-self.nI       # Current edge (boundary numbering)
                    e_sign = q_sign[iq]
                    visited[ie] = True
                    e_nbors = nbors[ie*self.nP:ie*self.nP+numNbors[ie]] # Neighboring partitions

                    # Loop through the neighbors of e, looking for next possible edge
                    for e2 in self.eemap[self.eemap_size[e]:self.eemap_size[e+1]]: 
                        ie2 = e2-self.nI
                        if (e2 >= self.nI and not visited[ie2]):
                            e2_nbors = nbors[ie2*self.nP:ie2*self.nP+numNbors[ie2]]
                            sameObject = (numNbors[ie2] == numNbors[ie]) and np.all(e_nbors == e2_nbors)
                            if (sameObject): # If in the same object
                                q[iq] = e2
                                self.obj_dofs[self.obj_size[iO+1]] = e2 # Put new edge in the object

                                # Select edge sign + get bridge node between e and e2
                                if(self.edgeNodes[e,0] == self.edgeNodes[e2,1]):
                                    q_sign[iq] = e_sign 
                                    self.obj_sign[self.obj_size[iO+1]] = e_sign
                                    self.obj_node[ie ,0] = iNode
                                    self.obj_node[ie2,1] = iNode
                                    bridgeNode = self.edgeNodes[e,0]
                                elif (self.edgeNodes[e,1] == self.edgeNodes[e2,0]):
                                    q_sign[iq] = e_sign 
                                    self.obj_sign[self.obj_size[iO+1]] = e_sign
                                    self.obj_node[ie ,1] = iNode
                                    self.obj_node[ie2,0] = iNode
                                    bridgeNode = self.edgeNodes[e,1]
                                elif (self.edgeNodes[e,0] == self.edgeNodes[e2,0]):
                                    q_sign[iq] = -e_sign
                                    self.obj_sign[self.obj_size[iO+1]] = -e_sign
                                    self.obj_node[ie ,0] = iNode
                                    self.obj_node[ie2,0] = iNode
                                    bridgeNode = self.edgeNodes[e,0]
                                else: # (self.edgeNodes[e,1] == self.edgeNodes[e2,1])
                                    q_sign[iq] = -e_sign
                                    self.obj_sign[self.obj_size[iO+1]] = -e_sign
                                    self.obj_node[ie ,1] = iNode
                                    self.obj_node[ie2,1] = iNode
                                    bridgeNode = self.edgeNodes[e,1]

                                i_com = np.where(self.com_loc == e2)[0][0]
                                self.obj_id1[iO] = min(self.obj_id1[iO],self.com_glob[i_com])
                                if (self.obj_id1[iO] == self.com_glob[i_com]): index_id1 = self.obj_size[iO+1]
                                self.obj_size[iO+1] += 1
                                iNode += 1
                                iq = iq + 1
                if (self.obj_sign[index_id1] == -1): self.obj_sign[self.obj_size[iO]:self.obj_size[iO+1]] = - self.obj_sign[self.obj_size[iO]:self.obj_size[iO+1]]
                iO += 1
                self.obj_size[iO+1] = self.obj_size[iO]

        self.nO = iO



    def getConstraints(self):
        # Get neighboring processors for every boundary edge
        numNbors = np.zeros(self.nB,dtype=int)
        for iP in range(self.nP):
            for iE in range(self.com_size[iP],self.com_size[iP+1]):
                e = self.com_loc[iE]-self.nI
                numNbors[e] += 1

        # Number of constraints
        self.nC = 0 
        self.obj_cts = np.zeros(self.nO, dtype=int)
        for iO in range(self.nO):
            # Pathological case: The object is a single edge
            if (self.obj_size[iO+1]-self.obj_size[iO] == 1): 
                self.obj_cts[iO] = 1
                self.nC += 1
            # Regular case: Chain length > 1
            else: 
                self.obj_cts[iO] = 2
                self.nC += 2

        # Create and fill C
        nC_max = 2*(self.obj_size[self.nO] + 2 * (self.nB - self.obj_size[self.nO]))
        iC = 0
        self.C_size = np.zeros(self.nC+1,dtype=int)
        self.C_idx  = np.zeros(nC_max,dtype=int)
        self.C      = np.zeros(nC_max)
        for iO in range(self.nO):
            # First Constraint
            nE      = self.obj_size[iO+1]-self.obj_size[iO]
            dofs_E  = self.obj_dofs[self.obj_size[iO]:self.obj_size[iO+1]]
            sign_E  = self.obj_sign[self.obj_size[iO]:self.obj_size[iO+1]]

            self.C_size[iC+1] = self.C_size[iC] + nE 
            self.C_idx[self.C_size[iC]:self.C_size[iC+1]] = dofs_E
            self.C[self.C_size[iC]:self.C_size[iC+1]]     = sign_E
            iC += 1

            # Second Constraint
            if (self.obj_cts[iO] == 2):
                nF = 0
                dofs_F = None
                if (self.dim == 3): # Only faces in 3D case
                    nF_max = 0 
                    for e in dofs_E : nF_max += self.eemap_size[e+1] - self.eemap_size[e]

                    dofs_F = np.zeros(nF_max,dtype=int)
                    for e in dofs_E: # For each edge in E
                        ie = e - self.nI
                        for e2 in self.eemap[self.eemap_size[e]:self.eemap_size[e+1]]: # Loop through neighbors
                            ie2 = e2 - self.nI
                            if (ie2 >= 0 and numNbors[ie2]+1 == 2): # If neighbor == face edge
                                self.obj_node[ie2 ,:] = -1
                                # Select only if they share interior node, and save which 
                                # node they share.
                                if (self.obj_node[ie ,0] != -1):
                                    if (self.edgeNodes[e2,0] == self.edgeNodes[e,0]): 
                                        self.obj_node[ie2 ,0] = self.obj_node[ie ,0]
                                        dofs_F[nF] = e2
                                        nF += 1
                                    elif (self.edgeNodes[e2,1] == self.edgeNodes[e,0]): 
                                        self.obj_node[ie2 ,1] = self.obj_node[ie ,0]
                                        dofs_F[nF] = e2
                                        nF += 1
                                if (self.obj_node[ie ,1] != -1):
                                    if (self.edgeNodes[e2,0] == self.edgeNodes[e,1]): 
                                        self.obj_node[ie2 ,0] = self.obj_node[ie ,1]
                                        dofs_F[nF] = e2
                                        nF += 1
                                    elif (self.edgeNodes[e2,1] == self.edgeNodes[e,1]): 
                                        self.obj_node[ie2 ,1] = self.obj_node[ie ,1]
                                        dofs_F[nF] = e2
                                        nF += 1

                # Constraint in the new basis 
                Cnew = np.ones(nE+nF)
                Cnew[0]   = 0.0
                Cnew[nE:] = 0.0

                # Change of basis
                Gt = np.zeros((nE+nF,nE+nF))
                Gt[0,:nE] = sign_E  # Phi^t
                for j in range(nE): # G_EN^t
                    ie = dofs_E[j] - self.nI
                    if (self.obj_node[ie ,0] != -1): Gt[self.obj_node[ie ,0]+1,j] = -1.0
                    if (self.obj_node[ie ,1] != -1): Gt[self.obj_node[ie ,1]+1,j] =  1.0
                for j in range(nF): # G_FN^t
                    ie = dofs_F[j] - self.nI
                    if (self.obj_node[ie ,0] != -1): Gt[self.obj_node[ie ,0]+1,j+nE] = -1.0
                    if (self.obj_node[ie ,1] != -1): Gt[self.obj_node[ie ,1]+1,j+nE] =  1.0
                    if (self.obj_node[ie ,0] != -1 and self.obj_node[ie ,1] != -1): print('>>>>>> WARNING!!!')
                for j in range(nF): # G_FF^t
                    Gt[j+nE,j+nE] = 1.0

                self.C_size[iC+1] = self.C_size[iC] + nE + nF # Cardinal(E) + Cardinal(F)
                self.C_idx[self.C_size[iC]:self.C_size[iC]+nE]   = dofs_E
                if (self.dim == 3): self.C_idx[self.C_size[iC]+nE:self.C_size[iC+1]] = dofs_F[:nF]
                self.C[self.C_size[iC]:self.C_size[iC+1]]        = np.linalg.solve(Gt,Cnew)
                iC += 1


    def initFine(self):
        # Invert local problem
        self.Aii = self.A[:self.nI,:self.nI]
        self.invAii  = scipy.sparse.linalg.factorized(self.Aii)
        self.Aib     = self.A[:self.nI,self.nI:]

        self.C_csr = csr((self.C,self.C_idx,self.C_size),shape=(self.nC,self.nI+self.nB))
        Aaux = bmat([[self.A           ,  self.C_csr.transpose() ] ,
                     [self.C_csr       ,  None                   ] ])
        self.invFine = scipy.sparse.linalg.factorized(Aaux)

        # Get local eigenvectors from Newmann problem
        self.Phi    = np.zeros((self.nB+self.nI, self.nC))
        self.Lambda = np.zeros((self.nC, self.nC))
        for j in range(self.nC):
            x = np.zeros(self.nI + self.nB + self.nC)
            x[self.nI + self.nB + j] = 1.0
            y = self.invFine(x)

            self.Phi[:,j]    = y[:self.nI+self.nB]
            self.Lambda[:,j] = y[self.nI+self.nB:]

        # Get weights 
        self.W = np.ones((self.n,1))
        for iP in range(self.nP):
            for iE in range(self.com_size[iP],self.com_size[iP+1]):
                self.W[self.com_loc[iE]] += 1
        for iE in range(self.n):
            self.W[iE] = 1.0 / self.W[iE]
