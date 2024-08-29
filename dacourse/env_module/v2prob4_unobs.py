# problem 4a
# build gaussian model
#%%
#%cd environ_module

#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lalg
import scipy.stats
import tqdm
a = np.array(range(12)).reshape((3,4))


# %%
# load OceanFlow data from csv files
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
# save time series data for location 100,100
# loc_x = 100
# loc_y = 60
loc_x = 150
loc_y = 410

ts_u = d4[:,0,loc_y,loc_x]
ts_v = d4[:,1,loc_y,loc_x]

train_idxs = np.array(range(len(ts_u)))
ts_u = ts_u[train_idxs]
ts_v = ts_v[train_idxs]

# get mean for each time by averaging neighbors and sliding window of time

# ts_u_mean = np.array([])
# ts_v_mean = np.array([])
ts_u_mean = np.zeros(len(ts_u))
ts_v_mean = np.zeros(len(ts_u))
# t_win = 3 # time sliding window for mean
# for t in range(len(ts_u_mean)):
#     u_mean = np.mean(d4[t-t_win:t+t_win,0,loc_y-2:loc_y+2,loc_x-2:loc_x+2])
#     v_mean = np.mean(d4[t-t_win:t+t_win,1,loc_y-2:loc_y+2,loc_x-2:loc_x+2])
#     ts_u_mean[t] = u_mean
#     ts_v_mean[t] = v_mean
# ts_u_mean[np.isnan(ts_u_mean)] = 0
# ts_v_mean[np.isnan(ts_v_mean)] = 0
# decided to use zero for mean instead
ts_v_mean[:] = 0
ts_u_mean[:] = 0

#%%
# Lets see how the data looks
x_hrs = train_idxs
x_hrs = x_hrs * 3*24
print(f' time series for one location: ts_u shape {ts_u.shape}')
plt.plot(x_hrs, ts_u, 'x-', label='u component')
plt.plot(x_hrs, ts_v, 'x-', label='v component')
plt.plot(x_hrs, ts_u_mean,  label='u mean')
plt.plot(x_hrs, ts_v_mean,  label='v mean')
plt.title('Flow data over time for location ' + str([loc_x,loc_y]))
plt.xlabel('Hours')
plt.legend()


# %%
# Partition observed data (training)
ind_x2 = train_idxs
np.random.seed(1)
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
theta1 = .1 #@param {type:"slider", min:0.1, max:10, step:0.1}
theta2 = 80 #@param {type:"slider", min:0.1, max:10, step:0.1}
theta3 = 0.001 #@param {type:"slider", min:0, max:4, step:0.01}

# covs = train_idxs[:,None] - train_idxs[None,:]
covs = x_hrs[:,None] - x_hrs[None,:]


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
plt.plot(x_hrs[k1_x2], mu_1_2, c='b', label="Conditional Mean P1 given P2")
plt.scatter(x_hrs[k1_x2], ts_u[k1_x2], c='orange', marker='o', label="P1 Measured Data")

plt.xlabel("Time")
plt.ylabel("Flow")

sigma_diags = np.diagonal(sigma_1_2)
plt.plot(x_hrs[k1_x2], mu_1_2 + 2*np.sqrt(sigma_diags), c='r', label="2 Sigmas above the mean")
plt.plot(x_hrs[k1_x2], mu_1_2 - 2*np.sqrt(sigma_diags), c='k', label="2 Sigmas below the mean")

plt.legend()
plt.show()



# %%
# Define Find Optimal Parameters function
def findOptim(a_vals, ell_vals, ind_x2, data, subTitle, tau):
    HOURS_PER_TICK = 3*24

    K = 5
    np.random.seed(10) # set seed to same value to generate same random vals
    perm = np.random.permutation(ind_x2.shape[0])
    ks_x2 = [ ind_x2[a] for a in map(np.sort, np.array_split(perm, K)) ]


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
                mu_1[:] = 0
                mu_2[:] = 0

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

    params_best_a = params_best['a']
    params_best_ell = params_best['ell']
    params_best_xs = params_best['xs']
    params_best_x_hrs = params_best_xs * HOURS_PER_TICK

    sigma_opt = params_best['a'] * np.exp( - covs**2 / params_best['ell']**2 )
    params_best['sigma_opt'] = sigma_opt

    plt.scatter(params_best_x_hrs, params_best['data'], c='b', marker='o', label="Data")
    plt.scatter(params_best_x_hrs, params_best['mu'], c='orange', marker='x', label=r"Predicted $\mu$")
    plt.plot(params_best_x_hrs, params_best['mu'] + 2*params_best['sigma'], c='r', label="2 sigma band")
    plt.plot(params_best_x_hrs, params_best['mu'] - 2*params_best['sigma'], c='r')
    plt.title( "{}: \nBest parameters: A={a:.3f}, L={ell:.3f}".format(subTitle, **params_best) )
    plt.xlabel("Time")
    plt.ylabel("Flow")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    pos = ax.imshow(logpdfs, origin='lower', aspect='auto', extent=[ell_vals[0], ell_vals[-1], a_vals[0], a_vals[-1]] )
    fig.colorbar(pos, ax=ax)
    ax.plot(params_best_ell, params_best_a, 'rx')
    ax.set_title(subTitle + ': \nAvg Log PDF over A,L parameter space')
    ax.set_xlabel('L param')
    ax.set_ylabel('A param')
    plt.show()
    return params_best



# %%
# Set Search Space, Find Optimal Parameters
a_vals = np.arange(.001, 4, 0.05)
ell_vals = np.arange(30, 360, 5)
tau = .001

ind_x2 = train_idxs
data = np.array([ts_u, ts_v, ts_u_mean, ts_v_mean])

# data[0] is data, data[2] is means
params_best_u = findOptim(a_vals, ell_vals, ind_x2, data, "Flow U Component", tau)


#%%
data = np.array([ts_v, ts_v, ts_v_mean, ts_v_mean])
params_best_v = findOptim(a_vals, ell_vals, ind_x2, data, "Flow V Component", tau)


############################################################################

# %%
# Estimate for the Unobserved

# changes:
# ind_x2 is obs ticks, one every 3 ticks if each tick is one day (24 hours)
# total x ticks 100*3 = 300 days
#  300 ticks instead of 100 ticks before

# ind_x1 is unobs ticks, between the obs ticks
# recalc sigma_opt for new x ticks 
#   - new cov matrix filled in for unobs and obs

# data in ts_u ts_v
n_xs = len(ts_u)*3 # add 2 ticks between each tick
x_idx_all = np.array(range(n_xs))
ind_x2 = np.array(range(0, n_xs, 3)) # obs idxs, 1 every 3
ind_x1 = np.array(sorted(list(set( x_idx_all ) - set(ind_x2))))

# cov and sigma
x_hrs = x_idx_all * 24 # 24 hrs per tick
covs = x_hrs[:,None] - x_hrs[None,:]   # shape 300 x 300
alldata = np.zeros((1,n_xs))


#%%
# Define estim_unobs()
def estim_unobs(params_best, alldata, covs, ind_x2, ind_x1, subTitle, doPlot):
    n_xs = covs.shape[0]
    # data[0, ind_x2]  should be u or v data for idxs out of 300


    sigma_opt = params_best['a'] * np.exp( - covs**2 / params_best['ell']**2 )

    # First  separate the components of the covariance matrix
    sigma_11_n = sigma_opt[ind_x1[:,None],ind_x1];
    sigma_12_n = sigma_opt[ind_x1[:,None],ind_x2];
    sigma_21_n = sigma_opt[ind_x2[:,None],ind_x1];
    sigma_22_n = sigma_opt[ind_x2[:,None],ind_x2];

    # mu_x1 = data[2, ind_x1]
    # mu_x2 = data[2, ind_x2]
    mu_x1 = np.zeros(len(ind_x1))
    mu_x2 = np.zeros(len(ind_x2))

    tau = .001

    sigma_22_n_noise = sigma_22_n + tau * np.eye(mu_x2.shape[0])

    x2 = alldata[0, ind_x2]

    mu_x1_n = mu_x1 + sigma_12_n.dot( lalg.solve(sigma_22_n_noise, x2 - mu_x2 ) )
    mu_x2_n = mu_x2 + sigma_22_n.dot( lalg.solve(sigma_22_n_noise, x2 - mu_x2 ) )

    sigma_x1_n = sigma_11_n - sigma_12_n.dot(lalg.inv(sigma_22_n_noise).dot(sigma_12_n.T))
    sigma_x2_n = sigma_22_n - sigma_22_n.dot(lalg.inv(sigma_22_n_noise).dot(sigma_22_n.T))

    new_mu = np.zeros(n_xs)
    new_mu[ind_x1] = mu_x1_n
    new_mu[ind_x2] = mu_x2_n

    new_sigma = np.zeros(n_xs)
    new_sigma[ind_x1] = np.sqrt(np.diagonal(sigma_x1_n))
    new_sigma[ind_x2] = np.sqrt(np.diagonal(sigma_x2_n))

    #xs = np.arange(1, 361)
    xs = np.arange(0, n_xs)

    if doPlot:
        plt.figure(figsize=(12,8))
        plt.plot(xs, new_mu, c='r', label=r"Predicted conditional mean", lw=0.5)
        plt.plot(xs, new_mu + 3*new_sigma, c='k', label="Three Sigma Upper Bound", lw=0.5)
        plt.plot(xs, new_mu - 3*new_sigma, c='b', label="Three Sigma Lower Bound", lw=0.5)
        plt.scatter(ind_x2+1, alldata[0, ind_x2], c='r', marker='o', label="Observed data", lw=0.5)
        # plt.scatter(ind_x1+1, alldata[0, ind_x1], c='r', marker='x', label="Observation of $X_1$", lw=0.5)
        plt.xlabel("Time")
        plt.ylabel("Flow")
        plt.title("Estimating Unobserved Data: " + subTitle)
        plt.legend()
        plt.show()
    return new_mu


#%%
# Run estim_unobs for U and V

# def estim_unobs(params_best, alldata, covs, ind_x2, ind_x1, subTitle):

alldata[0, ind_x2] = ts_u
new_mu = estim_unobs(params_best_u, alldata, covs, ind_x2, ind_x1, "U Comp " + f", location {[loc_x,loc_y]}", True)


alldata[0, ind_x2] = ts_v
new_mu = estim_unobs(params_best_v, alldata, covs, ind_x2, ind_x1, "V Comp" + f", location {[loc_x,loc_y]}", True)

# %%
# Generate unobs predictions for all locations

# alldata = np.zeros((1,n_xs))
# ts_u = d4[:,0,loc_y,loc_x]
# ts_v = d4[:,1,loc_y,loc_x]
def getUV(d4In, loc_x, loc_y):
    u = d4In[:,0, loc_y, loc_x]
    v = d4In[:,1, loc_y, loc_x]
    return (u,v)


newData4 = np.zeros( (n_xs, 2, d4.shape[2], d4.shape[3]) )

doPlot = False
dataIn = np.zeros( (1,n_xs) )
for loc_x in range( d4.shape[3] ):
    for loc_y in range( d4.shape[2] ):
        (u, v) = getUV(d4, loc_x, loc_y)
        
        if not all(u==0):
            dataIn[0, ind_x2] = u
            new_mu = estim_unobs(params_best_u, dataIn, covs, ind_x2, ind_x1, "U Comp " + f", location {[loc_x,loc_y]}", doPlot)
            newData4[:, 0, loc_y, loc_x] = new_mu

        if not all(v==0):
            dataIn[0, ind_x2] = v
            new_mu = estim_unobs(params_best_v, dataIn, covs, ind_x2, ind_x1, "V Comp" + f", location {[loc_x,loc_y]}", doPlot)
            newData4[:, 1, loc_y, loc_x] = new_mu
    if loc_x % 50 == 0:
        print( f' process loc_x {loc_x}')  # print progress
print( f' newData4 shape {newData4.shape}' )

# %%
