
import scipy
import numpy as np

import matplotlib.pyplot as plt


x = np.arange(20)/2

y = scipy.stats.norm.cdf(x, loc=5, scale=1)
print(x,y)
plt.plot(x,y)
np.hstack((x,y))
