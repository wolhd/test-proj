import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# create sample df with

# type, name, start, up
# interc, 'interc', 1000, none
# emi, 'e1', none, 800
# emi, 'e2', none, 900
# emi, 'e3', none, 1100

day = 24*3600
li = []
li.append(dict(type='interc',name='in',start=day+1000, up=None))
li.append(dict(type='emi',name='em1',start=None, up=day+800))
li.append(dict(type='emi',name='em2',start=None, up=day+900))
li.append(dict(type='emi',name='em3',start=None, up=day+1200))

df =pd.DataFrame(li,columns=['type','name','start','up'])

df['startdt'] = df['start'].apply(lambda x: None if np.isnan(x) else datetime.fromtimestamp(x))
df['updt'] = df['up'].apply(lambda x: None if np.isnan(x) else datetime.fromtimestamp(x))

# ways to append rows

li = [['a',1],['b',2]]
cols = ['c1','c2']
df2 = pd.DataFrame(li,columns=cols)
df2a = pd.DataFrame([['c',3]],columns=cols)
df3 = pd.concat([df2,df2a])
df3.loc[len(df3.index)] = ['d',4]

# print(df3)   


# loop through df and create new df

# new df
ndf = pd.DataFrame({},columns=['name','delta'])

# get interc row
interdf = df.loc[df['type'] == 'interc']
#print(interdf, len(interdf))
interStart = interdf.at[0,'startdt'].timestamp()

for (idx,row) in df.loc[df['type'] == 'emi'].iterrows(): #itertuples
    delta = row['updt'].timestamp() - interStart
    ndf.loc[len(ndf.index)] = [row['name'],delta]
    
print(ndf)

def create_row_series(bins, up, down):
    upIdx = np.digitize(up, bins)
    dnIdx = np.digitize(down, bins)
    row = np.zeros(len(bins))
    row[upIdx:dnIdx+1] = 1
    return row
    

min = ndf['delta'].min()
max = ndf['delta'].max()
speriod = 100
bins = np.arange(min,max+300+speriod,speriod)    
mat = np.empty((0,len(bins)), int)

for tu in ndf.itertuples():
    up = tu.delta
    down = up + 200
    row = create_row_series(bins, up, down)
    mat = np.append(mat, np.array([row]), axis=0)
    
print(mat)

line = mat.sum(axis=0)
plt.plot(bins,line, label='n1')
plt.plot(bins,line+1.2, label='n2')
plt.legend(loc="upper right")
plt.show()

#todo, read all pkl files of pattern into df's
