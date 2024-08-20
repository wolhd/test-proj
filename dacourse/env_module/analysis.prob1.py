#%%
import numpy as np

# %%

datadir = "OceanFLow/"
x = np.loadtxt(open(datadir + "1u.csv", "rb"), delimiter=",")

# %%
d1 = np.array([[1,2],[11,22]])
d2 = np.array([[3,4],[33,44]])
res = np.concatenate((d1,d2), axis=2)
print(np.shape(res))

print(res)

g=np.array([d1,d2])
np.concatenate((g, d1[np.newaxis,:,:]) , axis=0 )

# %%
# scratch on reading small csv's into 4-d array
for i in range(1,3):
    datadir = "sampleData/"
    ufilename = datadir + str(i) + "u.csv"
    vfilename = datadir + str(i) + "v.csv"
    udata = np.loadtxt(open(ufilename), delimiter=",")
    vdata = np.loadtxt(open(vfilename), delimiter=",")
    td3 = np.array([udata, vdata])
    if i == 1:
        td4 = np.array(td3[np.newaxis,:,:,:])
    else:
        td4 = np.concatenate((td4, td3[np.newaxis,:,:,:]))
print(f'td4 shape {np.shape(td4)}')

# dims:  time, u or v, y coord, x coord

#%%
for i in range(1,101):
    datadir = "OceanFlow/"
    ufilename = datadir + str(i) + "u.csv"
    vfilename = datadir + str(i) + "v.csv"
    udata = np.loadtxt(open(ufilename), delimiter=",")
    vdata = np.loadtxt(open(vfilename), delimiter=",")
    d3 = np.array([udata, vdata])
    if i == 1:
        d4 = np.array(d3[np.newaxis,:,:,:])
    else:
        d4 = np.concatenate((d4, d3[np.newaxis,:,:,:]))
# dims:  time, u or v, y coord, x coord
print(f'd4 shape: {np.shape(d4)}')

# %%
import matplotlib.pyplot as plt

#plt.hist(np.reshape(d4,(1,-1)) )
#%%
max_u = np.max(d4[:,0,:,:])
max_u_idx = np.unravel_index( np.argmax(d4[:,0,:,:]), d4.shape)

# %%
d3_u = d4[:,0,:,:]
max_u = np.max(d3_u)
max_u_idx = np.unravel_index( np.argmax(d3_u), d3_u.shape)
print(f' max {max_u}, idx {max_u_idx}')
timeIdx = max_u_idx[0]
y_idx = max_u_idx[1]
x_idx = max_u_idx[2]

# time spacing = 3 hrs
# xy grid = 3 km

print(f' max time {timeIdx*3} hours')
print(f' max x coord {x_idx*3} km')
print(f' max y coord {y_idx*3} km')



# %%
