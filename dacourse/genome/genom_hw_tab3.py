import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# tab 3
# clustering

X = np.load('gene_data/data/p1/X.npy')
Xproc = np.log2(X+1)

#------------------------------
pca = PCA(n_components=50)
pca_tr50 = pca.fit_transform(Xproc)

kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(pca_tr50)
labels = kmeans.labels_
plt.scatter(pca_tr50[:,0], pca_tr50[:,1], c=labels)
plt.show()

from sklearn.manifold import MDS
mds = MDS()
tr = mds.fit_transform(Xproc);
print(f' stress {mds.stress_}')
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(tr)
labels = kmeans.labels_

plt.scatter(tr[:,0],tr[:,1],c=labels)
plt.show()


from sklearn.manifold import TSNE
tsne = TSNE(perplexity=40, random_state=0)
tsneTr = tsne.fit_transform(pca_tr50)
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(tsneTr)
labels = kmeans.labels_
plt.scatter(tsneTr[:,0], tsneTr[:,1], c=labels)


cls = []
inertia = []
for i in range(11):
    cl = i+1
    kmeans = KMeans(n_clusters=cl, random_state=0, n_init="auto").fit(pca_tr50)
    labels = kmeans.labels_
    iner = kmeans.inertia_
    cls = cls + [cl]
    inertia = inertia + [iner]
    plt.scatter(pca_tr50[:,0], pca_tr50[:,1], c=labels)
    plt.show()

plt.plot(cls,inertia)
plt.show()

