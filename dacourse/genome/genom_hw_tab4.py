import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

X = np.load('gene_data/data/p1/X.npy')
Xproc = np.log2(X+1)

#------------------------------


kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(Xproc)
labels = kmeans.labels_
cen = kmeans.cluster_centers_

pca = PCA(n_components=2)
pcatr = pca.fit_transform(cen)

plt.scatter(pcatr[:,0],pcatr[:,1])
plt.show()

mds = MDS()
tr = mds.fit_transform(cen);

plt.scatter(tr[:,0],tr[:,1])
plt.show()

#-------------

pca = PCA(n_components=2)
pcatr = pca.fit_transform(X)
plt.scatter(pcatr[:,0],pcatr[:,1])
plt.show()

mds = MDS()
tr = mds.fit_transform(X);
plt.scatter(tr[:,0],tr[:,1])
plt.show()

tsne = TSNE(perplexity=40, random_state=0)
tsneTr = tsne.fit_transform(X)
plt.scatter(tsneTr[:,0], tsneTr[:,1])