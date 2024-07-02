import numpy as np

filedata = np.loadtxt('release_directed_graph.txt', dtype=int)

numNodes = filedata.max() + 1
adj = np.zeros([numNodes, numNodes])

for row in filedata:
    adj[row[0], row[1]] += 1

# how many nodes
uniq = np.unique(filedata)


import networkx as nx

g = nx.from_numpy_array(adj)

nx.draw(g)
sc = nx.simple_cycles(g)
# for s in sc:
#     print(f's in sc {s}')