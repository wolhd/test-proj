
import numpy as np
from scipy.linalg import issymmetric

# Graph Basics

# Tab 6

a = [[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
	[1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
	[0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
	[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
	[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]]
a = np.array(a)

issymmetric(a)

b = np.array([ [1,1],[0,0]])
issymmetric(b)

# what l for a^l is there where every cell is >0
al = a
for i in range(10):
    al = al @ a
    allnz = np.all(al)
    if allnz:
        print(f'l = {i+2}')
        break

import networkx as nx

g = nx.from_numpy_array(a)

nx.draw(g)

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
csr = csr_matrix(a)
ncomps, labels = connected_components(csr)
print(f'ncomps {ncomps}')

# a^l, l=5
a5 = a@a@a@a@a
print(a5)
