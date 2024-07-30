
import networkx as nx

import numpy as np

adj = np.array(	[[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
	[1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
	[0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
	[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
	[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])



G = nx.from_numpy_array(adj)

comm1 = {0,2,4,6,8}
comm2 = {1,3,5,7,9}
#nx.community.modularity(G, [{0, 1, 2}, {3, 4, 5}])
mo = nx.community.modularity(G, [ comm1, comm2 ])
print(f' modularity = {mo}')