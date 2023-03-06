import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)

for name, group in df.groupby(['A', 'B']):
    print(name)
    #print(group)
grps = [g for n,g in df.groupby('A')]    
pklfile = 'tmp.pkl'
grps[0].to_pickle(pklfile)

ndf = pd.read_pickle(pklfile)

import glob
flist = glob.glob('*e.txt')

print(flist)
