

#%%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv('CO2.csv', comment='"', header=[0], skiprows=2)
df = df.iloc[2:]
df = df.reset_index(drop=True)

# %%
df
# %%
