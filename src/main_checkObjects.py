###############################################################################
#####                             MAIN PROGRAM                            #####
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



def getGlobalOrdering(nP, fine):

    # Initialise
    for iP in range(nP):
        fine[iP].globalTag = np.zeros(fine[iP].n,dtype=int)

    counter = 0

    # Interior edges
    for iP in range(nP):
        nI = fine[iP].nI
        fine[iP].globalTag[:nI] = np.arange(counter,counter+nI)
        counter += nI

    # Boundary edges
    d = {}
    for iP in range(nP):
        nI = fine[iP].nI
        nB = fine[iP].nB
        for iB in range(nI,nI+nB):
            iB_com  = np.where(fine[iP].com_loc == iB)[0][0]
            iB_glob = fine[iP].com_glob[iB_com]

            if (iB_glob in d.keys()) :
                fine[iP].globalTag[iB] = d[iB_glob]
            else :
                fine[iP].globalTag[iB] = counter 
                d[iB_glob] = counter
                counter += 1

    # Get global tags for interior/boundary dofs only 
    for iP in range(nP):
        fine[iP].nodesI = fine[iP].globalTag[:fine[iP].nI]
        fine[iP].nodesB = fine[iP].globalTag[fine[iP].nI:]


############################################################################### 
###############################################################################


inputDir  = 'DOF166796'

inputPath = '../input/' + inputDir
caseName  = ''
nP = 0
if (inputDir == 'DOF40' or inputDir == 'DOF3564' ):
    caseName = 'wire-TRI03'
    nP = 4
elif (inputDir == 'DOF33062'):
    caseName = 'TS-TAPE-MIXED-ANG'
    nP = 8
elif (inputDir == 'DOF166796' ):
    caseName = 'TS-SLAB-HEX08-BULK'
    nP = 15
elif (inputDir == 'DOF669780' ):
    caseName = 'TS-SLAB-HEX08-BULK'
    nP = 23


# Read data from Alya
fine = []
print('> READING INPUTS')
for i in range(nP):
    fine.append(fineProc())
    IO.readFineProc(fine[i], inputDir, caseName, i)
    fine[i].getObjects()
    fine[i].getConstraints()

coarse = coarseProc()
coarse.createGlobalOrdering(nP,fine)
getGlobalOrdering(nP, fine)


# List of object Global ID's for each fine proc. 
Olist=np.empty(nP,dtype=object)
for iP in range(nP):
    Olist[iP] = coarse.com_obj[coarse.com_size[iP]:coarse.com_size[iP+1]]


# print('>> Object classification: ')
# for iP in range(nP):
#     print('   >> Proc ', iP, ':  ', Olist[iP])

# print('>> Object ID per proc: ')
# for iP in range(nP):
#     print('   >> Proc ', iP, ':  ')
#     print('      >> Edg = ', fine[iP].globalTag)
#     print('      >> ID1 = ', fine[iP].obj_id1[:fine[iP].nO])
#     print('      >> ID2 = ', fine[iP].obj_id2[:fine[iP].nO])
#     for iO in range(fine[iP].nO):
#         print('         >> ', fine[iP].obj_dofs[fine[iP].obj_size[iO]:fine[iP].obj_size[iO+1]])

print('>> Checking pathological cases: ')
if (fine[0].dim == 3):
    for iP in range(nP):
        # For each boundary dof, list of neighboring processors (inverse of the com_XXX arrays).
        numProcs = np.zeros(fine[iP].nB,dtype=int)
        procs    = -np.ones((fine[iP].nB*fine[iP].nP),dtype=int)
        for iP2 in range(fine[iP].nP):
            for e in range(fine[iP].com_size[iP2],fine[iP].com_size[iP2+1]):
                ie = fine[iP].com_loc[e]-fine[iP].nI
                procs[ie*fine[iP].nP + numProcs[ie]] = fine[iP].procs[iP2]
                numProcs[ie] += 1

        # For each local object, list of neighboring processors.
        objProcs = []
        for iO in range(fine[iP].nO):
            edges = fine[iP].obj_dofs[fine[iP].obj_size[iO]:fine[iP].obj_size[iO+1]]
            ie = edges[0] - fine[iP].nI
            objProcs.append(set(procs[ie*fine[iP].nP:ie*fine[iP].nP+numProcs[ie]]))

        # For each local object
        for iO in range(fine[iP].nO):
            edges = fine[iP].obj_dofs[fine[iP].obj_size[iO]:fine[iP].obj_size[iO+1]]
            signs = fine[iP].obj_sign[fine[iP].obj_size[iO]:fine[iP].obj_size[iO+1]]
            nodes = fine[iP].edgeNodes[edges,:]

            # For each edge in this local object
            for e in edges: 
                ie = e - fine[iP].nI
                pe = set(procs[ie*fine[iP].nP:ie*fine[iP].nP+numProcs[ie]])

                # A) Make sure all object edges have the same neighboring processors.
                if (objProcs[iO] != pe): 
                    print('>>> ERROR :: Obj ', iO, ' and edge ', e, ' have diff neighboring procs.')
                    print('     > pe   = ', pe)
                    print('     > pobj = ', objProcs[iO])

                # B) Make sure all connected face edges share a subset of the neighboring processors. 
                # C) Make sure no face edge is connected twice (weird things could hapen...)
                nbors = fine[iP].eemap[fine[iP].eemap_size[e]:fine[iP].eemap_size[e+1]]
                for e2 in nbors: 
                    ie2 = e2 - fine[iP].nI
                    if (ie2 >= 0 and numProcs[ie2] == 1): # If neighbor == face edge
                        connected = (fine[iP].obj_node[ie2 ,0] != -1) or (fine[iP].obj_node[ie2 ,1] != -1)
                        pe2 = set(procs[ie2*fine[iP].nP:ie2*fine[iP].nP+numProcs[ie2]])
                        if (connected and not pe2.issubset(pe)): 
                            print('>>> ERROR :: Different neighboring procs for obj ', iO, ' and edges [', e, ',', e2,']')
                            print('     > pe  = ', pe)
                            print('     > pe2 = ', pe2)

                        two_connected = (fine[iP].obj_node[ie2 ,0] != -1) and (fine[iP].obj_node[ie2 ,1] != -1)
                        if (two_connected): 
                            print('>>> WARNING :: Face edge ', e2, ' is double connected.')

            # D) If two objects share the same neighboring processors, make sure they are disconnected. 
            # WARNING:: This is incorrect, I think.... TODO: Revise this
            for iO2 in range(fine[iP].nO):
                if (iO2 != iO and objProcs[iO] == objProcs[iO2]):
                    edges2 = fine[iP].obj_dofs[fine[iP].obj_size[iO2]:fine[iP].obj_size[iO2+1]]
                    nodes2 = fine[iP].edgeNodes[edges2,:]
                    for dof in nodes: 
                        if (dof in nodes2): 
                                print('>>> ERROR :: Objects [', iO, ',', iO2,'] share nbors but are connected.')



print('>> Checking object global ordering: ')
for iO in range(coarse.nO):
    procs = []
    iprocs = []
    for iP in range(nP):
        if (iO in Olist[iP]):
            procs.append(iP)
            iprocs.append(np.where(Olist[iP] == iO)[0][0])

    edges = fine[procs[0]].obj_dofs[fine[procs[0]].obj_size[iprocs[0]]:fine[procs[0]].obj_size[iprocs[0]+1]]
    signs = fine[procs[0]].obj_sign[fine[procs[0]].obj_size[iprocs[0]]:fine[procs[0]].obj_size[iprocs[0]+1]]
    nodes = fine[procs[0]].edgeNodes[edges,:]

    perm = np.argsort(fine[procs[0]].globalTag[edges])
    g_edges = np.sort(fine[procs[0]].globalTag[edges])
    g_signs = signs[perm]
    g_nodes = nodes[perm,:]

    for iP in range(len(procs)):
        edges2 = fine[procs[iP]].obj_dofs[fine[procs[iP]].obj_size[iprocs[iP]]:fine[procs[iP]].obj_size[iprocs[iP]+1]]
        signs2 = fine[procs[iP]].obj_sign[fine[procs[iP]].obj_size[iprocs[iP]]:fine[procs[iP]].obj_size[iprocs[iP]+1]]
        nodes2 = fine[procs[iP]].edgeNodes[edges2,:]

        perm2 = np.argsort(fine[procs[iP]].globalTag[edges2])
        g_edges2 = np.sort(fine[procs[iP]].globalTag[edges2])
        g_signs2 = signs2[perm2]
        g_nodes2 = nodes2[perm2,:]

        if ((g_edges != g_edges2).any()): 
            print('>>> ERROR :: Different edges for object ', iO, ' and procs [', procs[0], ',', procs[iP],']')
            print('             ', g_edges, g_edges2)

        if ((g_signs != g_signs2).any()): 
            print('>>> ERROR :: Different signs for object ', iO, ' and procs [', procs[0], ',', procs[iP],']')
            print('             ', g_edges, g_signs)
            print('             ', g_edges2, g_signs2)
            print(g_nodes)
            print(g_nodes2)

