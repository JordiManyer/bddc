
import numpy as np
import math
from collections import deque

class object():
    def __init__(self): 
        self.nodes = []       # List of nodes 
        self.parts = []       # List of adjacent partitions
        self.type  = -1       #   -1 :: undefined, 0 :: vertex, 1:: edge, 2:: face


class mesh():
    def __init__(self, nodeAdjMap, partition):
        self.n = partition.size
        self.nodeMap = nodeAdjMap              # Node adjacency map, i.e nodeMap[i] = {Neighbors of node i}

        self.nP      = np.amax(partition) + 1  # Number of partitions
        self.parts   = partition               # Mesh partition
        self.nI      = np.zeros(self.nP)       # Number of interior DOFs for each partition
        self.nB      = 0                       # Number of DOFs in the interface

        self.nO      = 0                       # Number of objects
        self.objects = []                      # List of objects
        self.getObjects()

    def getObjects(self):
        # First loop: Catalog DOF data
        numNbors = np.zeros(self.n,dtype=int)
        bNodes   = deque()
        for i in range(self.n):
            if (self.parts[i] == -1): 
                bNodes.append(i)
                self.nB += 1
                for j in range(len(self.nodeMap[i])): 
                    if (self.parts[self.nodeMap[i][j]] == -1): 
                        numNbors[i] += 1 
            else:
                self.nI[self.parts[i]] += 1

        # Second loop: Group interface nodes into objects
        while (len(bNodes) != 0): # While not all have been visited
            obj = object()
            q = deque()

            q.append(bNodes.pop()) # Starting point
            while (len(q) != 0): # While nodes remain in the queue
                k = q.pop()
                if (self.parts[k] == -1): # If the node has not been visited

                    parts = [] # Neighboring partitions to this node
                    for j in range(len(self.nodeMap[k])): 
                        if (self.parts[self.nodeMap[k][j]] >= 0):
                            parts.append(self.parts[self.nodeMap[k][j]])

                    if (obj.nodes == []): # This is the first node in the object
                        obj.nodes.append(k)   # Add node to object
                        obj.parts = parts     # Add parts to object 
                        self.parts[k] = -2    # Mark the node as visited
                        for j in range(len(self.nodeMap[k])): # Add neighbors to queue as candidates
                            if (self.parts[self.nodeMap[k][j]] == -1):
                                q.append(self.nodeMap[k][j])
                        
                    elif (len(obj.parts) == len(parts)): # This is NOT the first node in the object
                        obj.nodes.append(k)   # Add node to object
                        self.parts[k] = -2    # Mark the node as visited
                        for j in range(len(self.nodeMap[k])): # Add neighbors to queue as candidates
                            if (self.parts[self.nodeMap[k][j]] == -1):
                                q.append(self.nodeMap[k][j])

            # When no more nodes remain in the queue, add object if not empty
            if (len(obj.nodes) != 0):
                self.nO += 1
                self.objects.append(obj)

        # Third loop: Object classification (faces/edges/corners)
        for i in range(self.nO):
            if   (len(self.objects[i].nodes) == 1): # corner
                self.objects[i].type = 0

                c = self.objects[i].nodes[0]
                self.parts[c] = -3
                for j in self.nodeMap[c]:
                    for k in self.nodeMap[j]:
                        if (self.parts[k] >=  0 and self.parts[k] not in self.objects[i].parts):
                            self.objects[i].parts.append(self.parts[k])

            else :                                  # edge
                self.objects[i].type = 1
        return


