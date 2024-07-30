# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:19:01 2024

@author: myname
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


X = np.load('gene_data/data/p1/X.npy')
y = np.load('gene_data/data/p1/y.npy')

print(f'X shape {X.shape}')
print(y.size)

# plt.plot(X[:,0])
# plt.show()

print(f' X nonzeros {X[X[:,0]>0,0]} ..')
firstcol = X[:,0]
fcnz = firstcol[firstcol>0]
print(f' first col nz {fcnz}')
# log2( x + 1)
sol2 = np.log2(fcnz+1)
print(f' sol2 {sol2}')

np.set_printoptions(precision=7)


# X is raw
# Xproc is processed
Xproc = np.log2(X+1)

'''

#

pca = PCA(n_components=380)
pcapr = PCA(n_components=380)
rawpca = pca.fit(X)
propca = pcapr.fit(Xproc)
raw_expl_ra = rawpca.explained_variance_ratio_
pro_expl_ra = propca.explained_variance_ratio_
print(f'raw expl var {rawpca.explained_variance_ratio_[0:6]}')
print(f'proc expl var {propca.explained_variance_ratio_[0:6]}')

print(f'raw expl ra cumsum {np.cumsum(raw_expl_ra)[-10:-1]}')
print(f'pro expl ra cumsum {np.cumsum(pro_expl_ra)[-10:-1]}')

raw_cs = np.cumsum(raw_expl_ra)
pro_cs = np.cumsum(pro_expl_ra)

raw_ind85 = np.where( (raw_cs>=.85) )[0][0]
pro_ind85 = np.where( (pro_cs>=.85) )[0][0]
print(f'raw ind85 {raw_ind85}, pro ind85 {pro_ind85}')
#

Xpr = Xproc

plt.scatter(Xpr[:,0], Xpr[:,1])

pc = PCA(n_components=2)
pc.fit(Xpr)

tr = pc.transform(Xpr)
plt.scatter(tr[:,0], tr[:,1])

#
'''
'''
from sklearn.manifold import MDS
mds = MDS()
tr = mds.fit_transform(Xproc);
print(f' stress {mds.stress_}')
plt.scatter(tr[:,0],tr[:,1])
plt.show()

from sklearn.manifold import TSNE

pca = PCA(n_components=50)
pcaTr = pca.fit_transform(Xproc)
tsne = TSNE(perplexity=40, random_state=np.random.randint(1,100))
tsneTr = tsne.fit_transform(pcaTr)
plt.scatter(tsneTr[:,0], tsneTr[:,1])
print(f' kl div {tsne.kl_divergence_}')
'''


