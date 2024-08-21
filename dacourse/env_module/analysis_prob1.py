#%%
import numpy as np

# %%

datadir = "OceanFlow/"
x = np.loadtxt(open(datadir + "1u.csv", "rb"), delimiter=",")

# %%
d1 = np.array([[1,2],[11,22]])
d2 = np.array([[3,4],[33,44]])
res = np.concatenate((d1,d2), axis=0)
print(np.shape(res))

print(res)

g=np.array([d1,d2])
gconc = np.concatenate((g, d1[np.newaxis,:,:]) , axis=0 )
print(np.shape(gconc))

# %%
# scratch on reading small csv's into 4-d array
# for i in range(1,3):
#     datadir = "sampleData/"
#     ufilename = datadir + str(i) + "u.csv"
#     vfilename = datadir + str(i) + "v.csv"
#     udata = np.loadtxt(open(ufilename), delimiter=",")
#     vdata = np.loadtxt(open(vfilename), delimiter=",")
#     td3 = np.array([udata, vdata])
#     if i == 1:
#         td4 = np.array(td3[np.newaxis,:,:,:])
#     else:
#         td4 = np.concatenate((td4, td3[np.newaxis,:,:,:]))
# print(f'td4 shape {np.shape(td4)}')

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
# flow is in km/hr

print(f' max time {timeIdx*3} hours')
print(f' max x coord {x_idx*3} km')
print(f' max y coord {y_idx*3} km')



# %%
d3_u = d4[:,0,:,:]
d3_v = d4[:,1,:,:]
print(f' shape d4 {d4.shape}')
print(f' u mean {np.mean(d3_u)}')
print(f' v mean {np.mean(d3_v)}')
print(f' u shape {d3_u.shape}')

#%%
# 3d array processing
# multi 2 3d arrays
td3_1 = np.reshape(np.array(range(1,25)), (2,3,4))
print(f' td3_1 shape {td3_1.shape}')
print(td3_1)

td3_2 = np.reshape(np.array(range(101,125)), (2,3,4))
print(f' td3_2 shape {td3_2.shape}')
print(td3_2)

# %%
# calc magnitude of flow for each x/ycoord and time
d3_mag = np.sqrt( np.square(d3_u) + np.square(d3_v) )
print(f' d3_mag shape {d3_mag.shape}')
d3_mag_var = np.var(d3_mag, axis=0)
print(f' d3_mag_var shape {d3_mag_var.shape}')

d3_mag_var[ d3_mag_var == 0 ] = np.nan
min_var = np.nanmin( d3_mag_var )
print(f' min var {min_var}')

min_idx = np.unravel_index( np.nanargmin( d3_mag_var ), d3_mag_var.shape )
print(f' min idx {min_idx}')

print(f' min var x coord {min_idx[1]*3} km , y coord {min_idx[0]*3} km')

# %%
# create image plot
def implot(data2d):
    fig, ax = plt.subplots()
    pos = ax.imshow(data2d, origin='lower', cmap='coolwarm')
    fig.colorbar(pos, ax=ax)
    plt.show()

implot(d3_u[30,:,:])

def savedebug(data, name):
    plt.plot(data.reshape(-1))
    plt.savefig('debug/' + name + '.png')


# %%

# Problem 2
# calc correlation coeff

# uses d3_u d3_v from above

n_t = d3_u.shape[0]
n_x = d3_u.shape[2]
n_y = d3_u.shape[1]

x_inds = list(range(0, n_x, 200))
y_inds = list(range(0, n_y, 200))

# for each x,y store x2,y2,corr of highest corr 
# [x_coord x y_coord x [y2,x2,corrcoef]]
hicorrmap = np.empty((n_y, n_x, 3)).astype(float)
hicorrmap[:] = np.nan
hicorrval = np.empty((n_y, n_x))
hicorrval[:] = np.nan

for y1 in y_inds:
    for x1 in x_inds:
        d1_u = d3_u[:,y1,x1]
        d1_v = d3_v[:,y1,x1]
        if all(d1_u==0) and all(d1_v==0):
            continue


        # corr vals for x1,y1 to all x2,y2's
        cmap = np.empty((n_y, n_x)).astype(float)
        cmap[:] = np.nan

        print(f' find corr for xy {x1},{y1}')
        # get corr for all x2,y2's
        for y2 in y_inds:
            for x2 in x_inds:
                if y2 == y1 and x2 == x1:
                    # this is corr of same
                    continue
                d2_u = d3_u[:,y2,x2]
                d2_v = d3_v[:,y2,x2]

                if all(d2_u==0) or all(d2_v==0):
                    continue
                
                d12_u = np.array([d1_u, d2_u])
                d12_v = np.array([d1_v, d2_v])
                corr_u = np.corrcoef(d12_u)[0,1]
                corr_v = np.corrcoef(d12_v)[0,1]
                # print(f' corr u {corr_u}')
                if np.isnan(corr_u) or np.isnan(corr_v):
                    continue

                cmap[y2,x2] = np.max([corr_u, corr_v])

        # find highest corr, save it as hi corr for x1,y1
        if np.all( np.isnan( cmap ) ):
            continue    

        savedebug(cmap, f'{x1}_{y1}')

        hicorr = np.nanmax(cmap)
        hicorrIdx = np.unravel_index( np.nanargmax(cmap), cmap.shape )
        hicorrmap[y1, x1, :] = np.array([hicorrIdx[0], hicorrIdx[1], hicorr])
        hicorrval[y1, x1] = hicorr
        print(f' hi corr val y1 x1 val {y1} {x1} {hicorr}')
    
nonnan = hicorrval[~np.isnan(hicorrval)]
plt.plot(nonnan)


# for u-component 

# %%
# example array
a = np.array(range(12)).reshape((3,4))

# %%
