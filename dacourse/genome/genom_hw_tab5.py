
import numpy as np

import matplotlib.pyplot as plt

X = np.load('gene_data/data/p2_unsupervised/X.npy')

print(X.shape)
Xpr = np.log2(X+1)

max = Xpr[:,0].max()
print(f'max col0 {max}')

