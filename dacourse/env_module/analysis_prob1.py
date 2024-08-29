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
from matplotlib.patches import Circle

# takes long time to run
#plt.hist(np.reshape(d4,(1,-1)))
#plt.title('all u v values histogram')
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
def implot(data2d, **kwargs):
    fig, ax = plt.subplots()
    pos = ax.imshow(data2d, origin='lower', **kwargs)
    fig.colorbar(pos, ax=ax)
    plt.show()

implot(d3_u[30,:,:])

def savedebugPlot(data, name):
    plt.plot(data.reshape(-1))
    plt.savefig('debug/' + name + '.png')
def savedebug(data, name):
    np.savetxt('debug/' + name + ".csv", data)

#%%
# create isnan map
n_x = d4.shape[3]
n_y = d4.shape[2]
nanmap = np.zeros((n_y, n_x))
nancount = 0
for x in range(n_x):
    for y in range(n_y):
        dt = d4[:,0,y,x]
        if np.all(np.isnan(dt)) or np.all(dt==0):
            nancount += 1
            continue
        dt = d4[:,1,y,x]
        if np.all(np.isnan(dt)) or np.all(dt==0):
            nancount += 1
            continue
        nanmap[y,x]=1.0
print(f' nan count {nancount}')
implot(nanmap, cmap='ocean')
# nanmap is 2d array of 1's 0's, if 0 no flow data, if 1 has flow data
#    shape 504,555

# %%

# Problem 2
# calc correlation coeff

# uses d3_u d3_v from above

n_t = d3_u.shape[0]
n_x = d3_u.shape[2]
n_y = d3_u.shape[1]

x_inds = list(range(0, n_x, 20))
y_inds = list(range(0, n_y, 20))

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

        # print(f' find corr for xy {x1},{y1}')
        # get corr for all x2,y2's
        for y2 in y_inds:
            for x2 in x_inds:
                if y2 == y1 and x2 == x1:
                    # this is corr of same
                    continue
                if np.linalg.norm([y2-y1, x2-x1]) < 200:
                    # too close in dist
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

                cmap[y2,x2] = np.nanmax([corr_u, corr_v])

        # find highest corr, save it as hi corr for x1,y1
        if np.all( np.isnan( cmap ) ):
            continue    

        #savedebug(cmap, f'{x1}_{y1}')

        hicorr = np.nanmax(cmap)
        hicorrIdx = np.unravel_index( np.nanargmax(cmap), cmap.shape )
        hicorrmap[y1, x1, :] = np.array([hicorrIdx[0], hicorrIdx[1], hicorr])
        hicorrval[y1, x1] = hicorr
        # print(f' hi corr={hicorr} x1,y1->x2,y2 {[x1,y1]} -> {[hicorrIdx[1],hicorrIdx[0]]}')
    
nonnan = hicorrval[~np.isnan(hicorrval)]
plt.plot(nonnan)


#%%
plt.hist(nonnan, bins=50)
plt.xlabel('correlation coefficient')
plt.ylabel('frequency')
plt.title('Highest correlation coeff for each sampled location histogram')

#%%
def mycorr(x1,y1,x2,y2, d4):
    label1 = f'{[x1,y1]}'
    label2 = f'{[x2,y2]}'
    d1_u = d4[:,0,y1,x1]
    d2_u = d4[:,0,y2,x2]
    d12_u = np.array([d1_u, d2_u])
    print(f' d12 u shape {d12_u.shape}')
    corr_u = np.corrcoef(d12_u)
    print(corr_u)
    d1_v = d4[:,1,y1,x1]
    d2_v = d4[:,1,y2,x2]
    d12_v = np.array([d1_v, d2_v])
    print(f' d12 v shape {d12_v.shape}')
    corr_v = np.corrcoef(d12_v)
    print(corr_v)
    fig, axs = plt.subplots(2)
    axs[0].plot(d12_u.T, label=[label1,label2])
    axs[0].set_ylabel('u flow')
    axs[1].plot(d12_v.T)
    axs[1].set_xlabel('time index')
    axs[1].set_ylabel('v flow')
    fig.suptitle(f'Flow values over time for location idxs {[x1,y1]} and {[x2,y2]}')

#%%
# get highest corr's
# 10 highest  (get 20 since duplicates) 
hvs = np.sort( hicorrval[~np.isnan(hicorrval)] )[-10:]
h20 = hvs[0]
xs = np.array([[0,0]])
ys = np.array([[0,0]])
for x1 in range(n_x):
    for y1 in range(n_y):
        vec = hicorrmap[y1,x1]
        corr = vec[2]
        if ~np.isnan(corr) and corr > h20:
            line1xs = np.array([[x1, vec[1]]] )
            line1ys = np.array([[y1, vec[0]]] )
            xs = np.concatenate((xs,line1xs), axis=0)
            ys = np.concatenate((ys,line1ys), axis=0)
xs = xs[1:]
ys = ys[1:]
xs = xs.T.astype(int)
ys = ys.T.astype(int)

#%%
# plot multiple lines,
# plot(xs,ys)  xs= x's each col new line, ys= same

fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xs, ys, '*:')
plt.title('5 highest correlation coeffs')
plt.show()

# describe: used grid pts 10 spacing, check corr for dist>200
#%%
mycorr(xs[0,-1],ys[0,-1], xs[1,-1], ys[1,-1], d4)

#mycorr(xs[0,-2],ys[0,-2], xs[1,-2], ys[1,-2], d4)


# %%
# example array
a = np.array(range(12)).reshape((3,4))
a[0,3] = 100
a[0,2] = 200
np.sort(a.reshape(-1))[-4:]


##########################################################################
# %%
# Simulation
import math

x_km = 100 * 3
y_km = 100 * 3
dt_hr = .1 # hr
len_t_hr = 300

def getFlowIndex(km, hr, maxIdx):
    i = math.floor(km/3)
    if i > maxIdx:
        i = -1
    t = math.floor(hr/3)
    return (i,t)

def km2Idxs(xs_km):
   return np.floor(xs_km/3).astype(int)

#%%
def sim(x_km, y_km, len_t_hr, dt_hr):
    # calculate trajectory of particle, return idxs for the traj points
    # args: start location of sim, length of sim, time increment of sim
    xs_km = np.array([x_km])
    ys_km = np.array([y_km])

    for t_hr in range(0, len_t_hr, dt_hr):
        x_km = xs_km[-1]
        y_km = ys_km[-1]

        xIdx = getFlowIndex(x_km, t_hr, 554)
        yIdx = getFlowIndex(y_km, t_hr, 503)
        t = xIdx[1] # time index
        x = xIdx[0] # x index
        y = yIdx[0] # y index
        # print(f' t {t}')
        if x < 0 or y < 0:
            u = 0
            v = 0
        else:
            u = d4[t, 0, y, x ]
            v = d4[t, 1, y, x ]

        x_km = x_km + u * dt_hr
        y_km = y_km + v * dt_hr
        xs_km = np.append(xs_km, x_km)
        ys_km = np.append(ys_km, y_km)
        # print(x_km)
    return (km2Idxs(xs_km), km2Idxs(ys_km))



#xy_in = np.array([[100,100], [200,110], [350,150], [400,200], [220,190],[420,420],[290,440],[150,410]])

#%%
# Run sim for set of start points  [x,y]
xy_in_arr = np.array([[100,100], [110,200], [200,220], [150, 400], [420,100], [470,220], [400,400]])
xy_in_kms = xy_in_arr*3
xys_traj_list = list()
for i in range(len(xy_in_kms)):
    xy_in = xy_in_kms[i]

    xys = sim( xy_in[0], xy_in[1], 300, 1 )
    xys_traj_list.append( xys )


#%%
# Run sim for a random set of start points
np.random.seed(4)
start_xs = np.random.uniform(low=5, high=550, size=(100,))
start_ys = np.random.uniform(low=5, high=500, size=(100,))
xy_in_arr = np.array([start_xs, start_ys]).T
xy_in_kms = xy_in_arr*3
xys_traj_list = list()
for i in range(len(xy_in_kms)):
    xy_in = xy_in_kms[i]

    xys = sim( xy_in[0], xy_in[1], 300, 1 )
    xys_traj_list.append( xys )



#%%    
# Plot sim results
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
for i in range( len( xys_traj_list ) ):
    xys = xys_traj_list[i]
    ax.plot(xys[0], xys[1])
plt.title('Particle flow, After 300 hours')



# %%
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
for i in range( len( xys_traj_list ) ):
    xys = xys_traj_list[i]
    ax.plot(xys[0][0], xys[1][0], 'x')
plt.title('Particle flow: Initial state')
# %%
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
for i in range( len( xys_traj_list ) ):
    xys = xys_traj_list[i]
    ax.plot(xys[0][0:100], xys[1][0:100])
plt.title('Particle flow: After 100 hours')
# %%
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
for i in range( len( xys_traj_list ) ):
    xys = xys_traj_list[i]
    ax.plot(xys[0][0:200], xys[1][0:200])
plt.title('Particle flow: After 200 hours')
# %%
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
for i in range( len( xys_traj_list ) ):
    xys = xys_traj_list[i]
    ax.plot(xys[0][:], xys[1][:])
plt.title('Particle flow: After 300 hours')

# %%
# Prob 3.b toy plane problem
def plot_traj(xys_traj_list, endIdx_t, title, **kwargs):
    fig, ax = plt.subplots()
    pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
    for i in range( len( xys_traj_list ) ):
        xys = xys_traj_list[i]
        ax.plot(xys[0][:endIdx_t], xys[1][:endIdx_t], **kwargs)
    plt.title(title)


#%%

sigma2 = 2000
xy_in_kms = np.random.multivariate_normal([300,1050], [[sigma2, 0],[0, sigma2]], size=30)
xys_traj_list = list()
for i in range(len(xy_in_kms)):
    xy_in = xy_in_kms[i]
    xys = sim( xy_in[0], xy_in[1], 120, 1 )
    xys_traj_list.append( xys )
plot_traj(xys_traj_list, 48, 'Toy Plane 48 hrs, var=2000')
plot_traj(xys_traj_list, 72, 'Toy Plane 72 hrs, var=2000')
plot_traj(xys_traj_list, 120, 'Toy Plane 120 hrs, var=2000')

#%%
sigma2 = 15000
xy_in_kms = np.random.multivariate_normal([300,1050], [[sigma2, 0],[0, sigma2]], size=30)
xys_traj_list = list()
for i in range(len(xy_in_kms)):
    xy_in = xy_in_kms[i]
    xys = sim( xy_in[0], xy_in[1], 120, 1 )
    xys_traj_list.append( xys )
plot_traj(xys_traj_list, 48, 'Toy Plane 48 hrs, var=15000')
plot_traj(xys_traj_list, 72, 'Toy Plane 72 hrs, var=15000')
plot_traj(xys_traj_list, 120, 'Toy Plane 120 hrs, var=15000')

#%%
# single toy
xys_traj_list = list()
xys = sim( 350, 1050, 120, 1 )
xys_traj_list.append( xys )
plot_traj(xys_traj_list, 48, 'Toy Plane 48 hrs', color='red')
plot_traj(xys_traj_list, 72, 'Toy Plane 72 hrs', color='red')
plot_traj(xys_traj_list, 120, 'Toy Plane 120 hrs', color='red')



#%%
# to plane problem - prev 
x_km = 300
y_km = 1050
# plots for 48, 72, 120 hrs
xys = sim( x_km, y_km, 48, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
plt.title('Toy Plane: After 48 hours')
plt.show()
xys = sim( x_km, y_km, 72, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
plt.title('Toy Plane: After 72 hours')
plt.show()
xys = sim( x_km, y_km, 120, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
plt.title('Toy Plane: After 120 hours')
# %%
# toy plane problem
x_km = 275
y_km = 1025
startTxt = f'start pt {[x_km,y_km]}'
# plots for 48, 72, 120 hrs
xys = sim( x_km, y_km, 48, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
plt.title('Toy Plane: After 48 hours, ' + startTxt)
plt.show()
xys = sim( x_km, y_km, 72, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
plt.title('Toy Plane: After 72 hours')
plt.show()
xys = sim( x_km, y_km, 120, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
circ = Circle((100,350),13, fill=False, edgecolor='gray')
ax.add_patch(circ)
plt.plot(100,350,'x')
ax.plot(xys[0], xys[1], 'r:')
plt.title('Toy Plane: After 120 hours')
# %%
x_km = 400
y_km = 1150
startTxt = f'start pt {[x_km,y_km]}'
# plots for 48, 72, 120 hrs
xys = sim( x_km, y_km, 48, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
circ = Circle((100,350),50, fill=False, edgecolor='gray')
ax.add_patch(circ)
plt.plot(100,350,'x')
plt.title('Toy Plane: After 120 hours')
plt.title('Toy Plane: After 48 hours, ' + startTxt)
plt.show()
xys = sim( x_km, y_km, 72, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
circ = Circle((100,350),50, fill=False, edgecolor='gray')
ax.add_patch(circ)
plt.plot(100,350,'x')
plt.title('Toy Plane: After 120 hours')
plt.title('Toy Plane: After 72 hours')
plt.show()
xys = sim( x_km, y_km, 120, 1 )
fig, ax = plt.subplots()
pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
ax.plot(xys[0], xys[1], 'r:')
circ = Circle((100,350),50, fill=False, edgecolor='gray')
ax.add_patch(circ)
plt.plot(100,350,'x')
plt.title('Toy Plane: After 120 hours')
# %%
