#%%
import numpy as np
import pandas as pd
import datetime

df = pd.read_csv('T10YIE.csv')
df['datetime'] = pd.to_datetime(df.DATE)
df['year'] = df.datetime.dt.year
df['month'] = df.datetime.dt.month

#%%
feb_avg = df[df.month == 2].T10YIE.mean()
feb2013_avg = df.query('month==2 and year==2013').T10YIE.mean()
# T10YIE is in percent

# %%

# deannulaized BER = (BER + 1) ** (1/12) -1

BERt = feb2013_avg / 100  # from percent to fraction
deannBER = (BERt + 1) ** (1/12) - 1
deannBER_percent = deannBER * 100
print(f' deannulized BER percent {deannBER_percent}')

# %%
