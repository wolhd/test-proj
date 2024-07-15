import numpy as np

filedata = np.loadtxt('release_directed_graph.txt', dtype=int)

numNodes = filedata.max() + 1
adj = np.zeros([numNodes, numNodes])

for row in filedata:
    adj[row[0], row[1]] += 1

# how many nodes
uniq = np.unique(filedata)
print(f'num nodes {uniq.size}')

import networkx as nx

g = nx.from_numpy_array(adj)

nx.draw(g)
sc = nx.simple_cycles(g)
for s in sc:
    print(f'graph has at least one cycle')
    break

# question 4
if adj.trace() > 0:
    print(f'node self loop exists')
# ? not sure how to detect directed cycle that is not self loop

#%%
# question 5
# p = edge exists between node1 node2 (node2 could be node1)
# so if there are 100 nodes, num edges per node = p * 100
# or avg edges per node / 100 = p
# -> num 1's in adj mat /100 = avg edges per node
edge_cnt = np.count_nonzero(adj)
p =  edge_cnt / 100 / 100
# 1030 / 100 / 100 = 1030/10,000
print(f'edge cnt: {edge_cnt}, prob of directed edge on a node: p = {p}')
# %%

# Circle Graph
n = 4
k = 2
print(f'k = {k}')
for i in range(n):
    connect_to = (i + k) % n
    print(f'i = {i} -> {connect_to}')

# %%
