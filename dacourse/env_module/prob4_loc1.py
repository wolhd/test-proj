# problem 4a
# build gaussian model

#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lalg
import scipy.stats
import tqdm
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
loc_x = 120
loc_y = 300
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
ts_v_mean[:] = 0
ts_u_mean[:] = 0

#%%
# Lets see how the data looks
print(f' time series for one location: ts_u shape {ts_u.shape}')
plt.plot(ts_u, 'x-', label='u component')
plt.plot(ts_v, 'x-', label='v component')
plt.plot(ts_u_mean,  label='u mean')
plt.plot(ts_v_mean,  label='v mean')
plt.title('Flow (training data) over time for location ' + str([loc_x,loc_y]))
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
# Define Find Optimal Parameters function
def findOptim(a_vals, ell_vals, ind_x2, data, subTitle):
   
    K = 5
    np.random.seed(10) # set seed to same value to generate same random vals
    perm = np.random.permutation(ind_x2.shape[0])
    ks_x2 = [ ind_x2[a] for a in map(np.sort, np.array_split(perm, K)) ]

    tau = .001

    target_best = -np.inf
    params_best = {}

    logpdfs = np.zeros((len(a_vals), len(ell_vals)))

    a_idx = 0
    for a_param in tqdm.tqdm(a_vals):
        ell_idx = 0
        for ell_param in ell_vals:
            sigma = a_param * np.exp( - covs**2 / ell_param**2 )

            vals = []
            
            for k in range(K):
                x_test = ks_x2[k]
                x_train = np.sort(np.array(list(set(ind_x2) - set(x_test))))

                mu_1 = data[2, x_test]
                mu_2 = data[2, x_train]

                sigma_11 = sigma[x_test[:,None],x_test]
                sigma_12 = sigma[x_test[:,None],x_train]
                sigma_21 = sigma[x_train[:,None],x_test]
                sigma_22 = sigma[x_train[:,None],x_train]

                sigma_22_noise = sigma_22 + tau * np.eye(mu_2.shape[0])
                x2_2 = data[0, x_train]

                mu_1_2 = mu_1 + sigma_12.dot( lalg.solve(sigma_22_noise, x2_2 - mu_2 ) )
                sigma_1_2 = sigma_11 - sigma_12.dot(lalg.inv(sigma_22_noise).dot(sigma_12.T))

                val = scipy.stats.multivariate_normal.logpdf(data[0, x_test], mean=mu_1_2, cov=sigma_1_2)
                vals.append(val)

            total = sum(vals)
            avg = total / K
            logpdfs[a_idx, ell_idx] = avg

            if total > target_best:
                params_best = dict(
                    a=a_param,
                    ell=ell_param,
                    xs=x_test+1,
                    data=data[0, x_test],
                    mu=mu_1_2,
                    sigma=np.sqrt(np.diagonal(sigma_1_2))
                )
                target_best = total
            ell_idx += 1
        a_idx += 1


    sigma_opt = params_best['a'] * np.exp( - covs**2 / params_best['ell']**2 )
    plt.scatter(params_best['xs'], params_best['data'], c='b', marker='o', label="Data")
    plt.scatter(params_best['xs'], params_best['mu'], c='orange', marker='x', label=r"Predicted $\mu$")
    plt.plot(params_best['xs'], params_best['mu'] + 2*params_best['sigma'], c='r', label="2 sigma band")
    plt.plot(params_best['xs'], params_best['mu'] - 2*params_best['sigma'], c='r')
    plt.title( "{}: \nBest parameters: A={a:.3f}, L={ell:.3f}".format(subTitle, **params_best) )
    plt.xlabel("Time idx")
    plt.ylabel("Flow")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    pos = ax.imshow(logpdfs, origin='lower', extent=[ell_vals[0], ell_vals[-1], a_vals[0], a_vals[-1]] )
    fig.colorbar(pos, ax=ax)
    ax.set_title(subTitle + ': \nAvg Log PDF over A,L parameter space')
    ax.set_xlabel('L param')
    ax.set_ylabel('A param')
    plt.show()
    return logpdfs



# %%
# Set Search Space, Find Optimal Parameters
a_vals = np.arange(.01, 2, 0.02)
ell_vals = np.arange(.1, 4, 0.1)

ind_x2 = train_idxs
data = np.array([ts_u, ts_v, ts_u_mean, ts_v_mean])

# data[0] is data, data[2] is means
findOptim(a_vals, ell_vals, ind_x2, data, "Flow U Component")


#%%
data = np.array([ts_v, ts_v, ts_v_mean, ts_v_mean])
findOptim(a_vals, ell_vals, ind_x2, data, "Flow V Component")


# %%
