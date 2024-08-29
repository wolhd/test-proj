# New simulation using conditional probability predictionss
# to fill in 

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

#%%
# read in Flow csv data
datadir = "environ_module/OceanFlow/"
for i in range(1,101):
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
def implot(data2d, **kwargs):
    fig, ax = plt.subplots()
    pos = ax.imshow(data2d, origin='lower', **kwargs)
    fig.colorbar(pos, ax=ax)
    plt.show()

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
print(f' nanmap shape {nanmap.shape}')


##########################################################################
# %%
# def Simulation

def getFlowIndex(km, hr, maxIdx):
    i = math.floor(km/3)
    if i > maxIdx:
        i = -1
    t = math.floor(hr/3)
    return (i,t)

def km2Idxs(xs_km):
   return np.floor(xs_km/3).astype(int)

# Note: uses d4
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

def plot_traj(xys_traj_list, endIdx_t, title, **kwargs):
    fig, ax = plt.subplots()
    pos = ax.imshow(nanmap, origin='lower', cmap='ocean')
    for i in range( len( xys_traj_list ) ):
        xys = xys_traj_list[i]
        ax.plot(xys[0][:endIdx_t], xys[1][:endIdx_t], **kwargs)
    plt.title(title)


#%%
# Run sim
np.random.seed(4)

# single toy
xys_traj_list = list()
xys = sim( 350, 1050, 120, 1 )
xys_traj_list.append( xys )
# plot_traj(xys_traj_list, 48, 'Toy Plane 48 hrs', color='red')
# plot_traj(xys_traj_list, 72, 'Toy Plane 72 hrs', color='red')
plot_traj(xys_traj_list, 120, 'Toy Plane 120 hrs', color='red')