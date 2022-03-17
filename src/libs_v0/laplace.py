import numpy as np 
import scipy.sparse as sp

def laplace1D(n):
    A_dense = np.zeros((n,n))
    for i in range(n):
        if i != 0: A_dense[i,i-1] = -1
        A_dense[i,i]   =  2
        if i != n-1: A_dense[i,i+1] = -1
    return sp.csr_matrix(A_dense)

def laplace2D(nSide):
    diag=np.ones([nSide*nSide])
    mat=sp.spdiags([diag,-2*diag,diag],[-1,0,1],nSide,nSide)
    I=sp.eye(nSide)
    return sp.kron(I,mat,format='csr')+sp.kron(mat,I)