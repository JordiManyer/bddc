import numpy as np 
import math
import scipy.sparse.linalg
from scipy.sparse import csr_matrix as csr
from scipy.sparse import bmat

from fineProc import *
from coarseProc import *


class jacobi():
    def __init__(self, fineProcs):
        self.nP    = len(fineProcs)
        self.fine  = fineProcs

    def init(self):
        for iP in range(self.nP):                           # Done in parallel (local)
            self.fine[iP].getObjects()
            self.fine[iP].getConstraints()
            self.fine[iP].initFine()

    def apply(self, r):
        n = r.size
        r2 = r.flatten()
        z = np.zeros(n)
        for iP in range(self.nP):
            D = self.fine[iP].A.diagonal()
            z[self.fine[iP].globalTag] += r2[self.fine[iP].globalTag]/D
        return z.reshape((n,1))
        