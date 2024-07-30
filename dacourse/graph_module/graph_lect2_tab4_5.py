

import networkx as nx

import numpy as np


a = [[1,0,0,0],
    [1,0,0,0],
    [1,0,0,0],
    [1,0,0,0]]
adj = np.array(a)

G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
centrality = nx.eigenvector_centrality(G)
print(f' a centrality {sorted((v, f"{c:0.2f}") for v, c in centrality.items())}')


#%%
b = [[1,1,1,1],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]]
badj = np.array(b)
bG = nx.from_numpy_array(badj, create_using=nx.DiGraph)
bcentrality = nx.eigenvector_centrality(bG)
bsort = sorted((v, f"{c:0.2f}") for v, c in bcentrality.items())
print(f'b centrality {bsort}')

# %%

c = 	[[1,1,1,1],
    [1,0,0,0],
    [1,0,0,0],
    [1,0,0,0]]
cadj = np.array(c)
cG = nx.from_numpy_array(cadj, create_using=nx.DiGraph)
ccentr = nx.eigenvector_centrality(cG)
csort = sorted( (v, f"{c:0.6f}") for v, c in ccentr.items())
print(f'c centrality {csort}')

# %%

# determinant
d = [[0,1], [0,0]]
np.linalg.det(d)

# %%


# tab 5

kc = 	[[0,1,1,1],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]]
kc_adj = np.array(kc)
kc_G = nx.from_numpy_array(kc_adj, create_using=nx.DiGraph)

kc_cent = nx.katz_centrality(kc_G, alpha=0.1, beta=1, normalized=True)
print(f'kc centrality {kc_cent}')

# %%
