# problem 4a
# build gaussian model

#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lalg

a = np.array(range(12)).reshape((3,4))


# %%
# load OceanFlow data from csv files
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
# save time series data for location 100,100
loc_x = 100
loc_y = 100
ts_u = d4[:,0,loc_y,loc_x]
ts_v = d4[:,1,loc_y,loc_x]

train_idxs = np.array(range(90))
ts_u = ts_u[train_idxs]
ts_v = ts_v[train_idxs]

# get mean for each time by averaging neighbors and sliding window of time

# ts_u_mean = np.array([])
# ts_v_mean = np.array([])
ts_u_mean = np.zeros(len(ts_u))
ts_v_mean = np.zeros(len(ts_u))
t_win = 3 # time sliding window for mean
for t in range(len(ts_u_mean)):
    u_mean = np.mean(d4[t-t_win:t+t_win,0,loc_y-2:loc_y+2,loc_x-2:loc_x+2])
    v_mean = np.mean(d4[t-t_win:t+t_win,1,loc_y-2:loc_y+2,loc_x-2:loc_x+2])
    ts_u_mean[t] = u_mean
    ts_v_mean[t] = v_mean
ts_u_mean[np.isnan(ts_u_mean)] = 0
ts_v_mean[np.isnan(ts_v_mean)] = 0


#%%
# Lets see how the data looks
print(f' time series for one location: ts_u shape {ts_u.shape}')
plt.plot(ts_u, 'x-', label='u component')
plt.plot(ts_v, 'x-', label='v component')
plt.plot(ts_u_mean,  label='u mean')
plt.plot(ts_v_mean,  label='v mean')
plt.title('Flow data over time for one location (training data)')
plt.legend()


# %%
# Partition observed data (training)
ind_x2 = train_idxs
perm = np.random.permutation(ind_x2.shape[0])
k1_x2, k2_x2 = [ ind_x2[a] for a in map(np.sort, np.split(perm, 2)) ]

for k_x2, label, c in [(k1_x2, "Partition 1", 'b'), (k2_x2, "Partition 2", 'r')]:
  plt.scatter(k_x2, ts_u[k_x2], label=label, c=c, marker='x')

x = train_idxs
plt.plot(x, ts_u_mean, c='k', label="Mean")
plt.legend()
plt.title('Partition of u flow data')
plt.show()


# %%
# Condition Distribution of Partition1 given Partition 2
theta1 = .4 #@param {type:"slider", min:0.1, max:10, step:0.1}
theta2 = 1 #@param {type:"slider", min:0.1, max:10, step:0.1}
theta3 = 0.001 #@param {type:"slider", min:0, max:4, step:0.01}

covs = train_idxs[:,None] - train_idxs[None,:]

sigma = theta1 * np.exp( - covs**2 / theta2**2 )

sigma_11 = sigma[k1_x2[:,None],k1_x2]
sigma_12 = sigma[k1_x2[:,None],k2_x2]
sigma_21 = sigma[k2_x2[:,None],k1_x2]
sigma_22 = sigma[k2_x2[:,None],k2_x2]

mu_1 = ts_u_mean[k1_x2]
mu_2 = ts_u_mean[k2_x2]

sigma_22_noise = sigma_22 + theta3 * np.eye(mu_2.shape[0])
x2_2 = ts_u[k2_x2]

mu_1_2 = mu_1 + sigma_12.dot( lalg.solve(sigma_22_noise, x2_2-mu_2 ) )
mu_2_2 = mu_2 + sigma_22.dot( lalg.solve(sigma_22_noise, x2_2-mu_2 ) )

sigma_1_2 = sigma_11 - sigma_12.dot(lalg.inv(sigma_22_noise).dot(sigma_12.T))
sigma_2_2 = sigma_22 - sigma_22.dot(lalg.inv(sigma_22_noise).dot(sigma_22.T))


#%%
# Plot Condition Distribution of Partition1 given Partition 2
plt.plot(k1_x2, mu_1_2, c='b', label="Conditional Mean P1 given P2")
plt.scatter(k1_x2, ts_u[k1_x2], c='orange', marker='o', label="P1 Measured Data")

plt.xlabel("Time idx")
plt.ylabel("Flow")

sigma_diags = np.diagonal(sigma_1_2)
plt.plot(k1_x2+1, mu_1_2 + 2*np.sqrt(sigma_diags), c='r', label="2 Sigmas above the mean")
plt.plot(k1_x2+1, mu_1_2 - 2*np.sqrt(sigma_diags), c='k', label="2 Sigmas below the mean")

plt.legend()
plt.show()


# %%
# Define Search Space
a_vals = np.arange(70, 90, 0.1)
ell_vals = np.arange(2, 4, 0.01)

# %%
import scipy.stats
import tqdm

ind_x2 = np.arange(0, 360, 4)
ind_x2 = np.arange(0, 80, 4)

K = 5
perm = np.random.permutation(ind_x2.shape[0])
ks_x2 = [ ind_x2[a] for a in map(np.sort, np.array_split(perm, K)) ]
ks_x2
# %%
