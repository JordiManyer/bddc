
import numpy as np
import csv
from scipy.sparse import csr_matrix as csr


def readFineProc(fine, inputDir, caseName, i):
    inputPath = '../input/' + inputDir
    
    fine.rank = i
    if (inputDir == 'DOF166796' or inputDir == 'DOF669780'):
        fine.dim = 3

    # Read system
    A,b = readSystem(inputPath,i+1)
    fine.n = A.get_shape()[0]
    fine.A = A
    fine.b = b.reshape((fine.n,1))

    # Read communication
    if (inputDir == 'DOF40' or inputDir == 'DOF3564'):
        fine.nP, fine.procs, fine.com_size, fine.com_glob, fine.com_loc = readCom(inputPath,caseName,i+1)
    else:
        fine.nP, fine.procs, fine.com_size, fine.com_loc, fine.com_glob = readCom(inputPath,caseName,i+1)
    fine.nI = np.amin(fine.com_loc)
    fine.nB = fine.n-fine.nI

    # Read connectivity
    fine.elemEdge, fine.signs, fine.eemap_size, fine.eemap, fine.edgeNodes, fine.nemap_size, fine.nemap = readConnectivity(inputPath,i+1)
    fine.nE = fine.elemEdge.shape[0]
    fine.nN = fine.nemap_size.size - 1



def readSystem(folderName, proc):
    # Count DOFs
    filename = folderName + '/magnet_ia-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        nDOFs = sum(1 for line in csvfile) - 1

    # Read Matrix in CSR format
    rows = np.zeros(nDOFs+1,dtype=int)
    filename = folderName + '/magnet_ia-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            rows[i] = int(line[0])-1
            i += 1

    nnz = rows[nDOFs]
    cols = np.zeros(nnz,dtype=int)
    data = np.zeros(nnz)
    filename = folderName + '/magnet_ja-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            cols[i] = int(line[0])-1
            i += 1
    filename = folderName + '/magnet_aa-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            data[i] = float(line[0])
            i += 1

    A = csr((data,cols,rows), shape=(nDOFs,nDOFs))

    # Read Vector 
    b = np.zeros(nDOFs)
    filename = folderName + '/magnet_rhs-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            b[i] = float(line[0])
            i += 1

    return A, b






def readCom(folderName, caseName, proc):

    # Get Neighboring processor IDs 
    filename = folderName + '/' + caseName + '-neights-' + str(proc) + '.mtx'
    with open(filename, 'r') as csvfile:
        nProcs = sum(1 for line in csvfile) - 2

    procs = np.zeros(nProcs,dtype=int)
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        next(myreader)
        next(myreader)
        i = 0
        for line in myreader:
            procs[i] = int(line[1])-1
            i += 1

    # Get ghost structure for edges
    bedge_size = np.zeros(nProcs+1,dtype=int)
    filename = folderName + '/' + caseName + '-bedge_size-' + str(proc) + '.mtx'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        next(myreader)
        next(myreader)
        i = 0
        for line in myreader:
            bedge_size[i] = int(line[1])-1
            i += 1

    nBedge = bedge_size[nProcs]
    bedge_loc  = np.zeros(nBedge,dtype=int)
    bedge_glob = np.zeros(nBedge,dtype=int)
    filename = folderName + '/' + caseName + '-bedge_perm_loc-' + str(proc) + '.mtx'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        next(myreader)
        next(myreader)
        i = 0
        for line in myreader:
            bedge_loc[i] = int(line[2])-1
            i += 1

    filename = folderName + '/' + caseName + '-bedge_perm_glo-' + str(proc) + '.mtx'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        next(myreader)
        next(myreader)
        i = 0
        for line in myreader:
            bedge_glob[i] = int(line[2])-1
            i += 1


    # Get ghost structure for nodes
    bnode_size = np.zeros(nProcs+1,dtype=int)
    filename = folderName + '/' + caseName + '-bound_size-' + str(proc) + '.mtx'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        next(myreader)
        next(myreader)
        i = 0
        for line in myreader:
            bnode_size[i] = int(line[1])-1
            i += 1

    nBnode = bnode_size[nProcs]
    bnode_loc  = np.zeros(nBnode,dtype=int)
    bnode_glob = np.zeros(nBnode,dtype=int)
    filename = folderName + '/' + caseName + '-bound_perm_loc-' + str(proc) + '.mtx'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        next(myreader)
        next(myreader)
        i = 0
        for line in myreader:
            bnode_loc[i] = int(line[2])-1
            i += 1

    filename = folderName + '/' + caseName + '-bound_perm_glo-' + str(proc) + '.mtx'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        next(myreader)
        next(myreader)
        i = 0
        for line in myreader:
            bnode_glob[i] = int(line[2])-1
            i += 1

    return nProcs, procs, bedge_size, bedge_loc, bedge_glob
    





def readConnectivity(folderName, proc):
    # Read sizes
    filename = folderName + '/magnet_eledg-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        nElem = sum(1 for line in csvfile)
    nNbors = 0
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        for line in myreader:
            nNbors = max(nNbors,len(line))

    # Read element - edge connectivity
    ElemEdgeMap = -1 * np.ones((nElem,nNbors),dtype=int)
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            for j in range(len(line)):
                ElemEdgeMap[i,j] = int(line[j])-1
            i += 1

    # Read edge signs in each element
    filename = folderName + '/magnet_signs-' + str(proc) + '.dat'
    EdgeSignMap = np.zeros((nElem,nNbors))
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            for j in range(len(line)):
                EdgeSignMap[i,j] = float(line[j])
            i += 1

    # Read edge - edge connectivity
    filename = folderName + '/magnet_iee-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        nEdges = sum(1 for line in csvfile)-1

    edge_sizes = np.zeros(nEdges+1,dtype=int)
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            edge_sizes[i] = int(line[0])-1
            i += 1

    nnz = edge_sizes[nEdges]
    edge_data = np.zeros(nnz,dtype=int)
    filename = folderName + '/magnet_jee-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            edge_data[i] = int(line[0])-1
            i += 1

    # Read nodes for each edge
    edgeNodes = np.zeros((nEdges,2),dtype=int)
    filename = folderName + '/magnet_ien-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            edgeNodes[i,0] = int(line[0])-1
            edgeNodes[i,1] = int(line[1])-1
            i += 1

    # Read Node-Edge map
    filename = folderName + '/magnet_ine-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        nNodes = sum(1 for line in csvfile)-1

    ne_sizes = np.zeros(nNodes+1,dtype=int)
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            ne_sizes[i] = int(line[0])-1
            i += 1

    nnz = ne_sizes[nNodes]
    ne_data = np.zeros(nnz,dtype=int)
    filename = folderName + '/magnet_jne-' + str(proc) + '.dat'
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ', quotechar='|', skipinitialspace=True)
        i = 0
        for line in myreader:
            ne_data[i] = int(line[0])-1
            i += 1

    return ElemEdgeMap, EdgeSignMap, edge_sizes, edge_data, edgeNodes, ne_sizes, ne_data
