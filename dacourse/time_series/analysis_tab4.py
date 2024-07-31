
#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import datetime

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.deterministic import DeterministicProcess

# %%

# CPI data

# for training, use < Sept 2013

dfcpi = pd.read_csv('CPI.csv')
dfcpi = dfcpi[ dfcpi.CPI.notna() ]
plt.plot(dfcpi.CPI.values)

# %%
# create datetime objs
dfcpi['datetime'] = pd.to_datetime(dfcpi['date'])
dfcpi['year'] = dfcpi.datetime.dt.year
dfcpi['month'] = dfcpi.datetime.dt.month
dfcpi['day'] = dfcpi.datetime.dt.day
print(len(dfcpi))
# keep only one entry per month
dfcpi = dfcpi[ dfcpi.day == 24]
print(len(dfcpi))

#%%
dfcpi['monthIdx'] = (dfcpi.year.values - 2008)*12 + dfcpi.month.values
# start at 0
dfcpi.monthIdx = dfcpi.monthIdx.values - dfcpi.monthIdx.values[0]
plt.plot(dfcpi.monthIdx, dfcpi.CPI)
# %%

# split training data
train_prior_todate = datetime.datetime(2013,9,1)
dftrain = dfcpi[ dfcpi.datetime < train_prior_todate ]

# %%
# fit linear trend on training data
x = np.reshape(dftrain.monthIdx.values, [-1, 1])
y = np.reshape(dftrain.CPI.values, [-1, 1])
reg = LinearRegression().fit(x, y)

y_hat = reg.predict(x)
plt.plot(x, y, label='original data')
plt.plot(x, y_hat, 'r', label='fitted line')
plt.legend()
print(f'intercept {reg.intercept_} coef {reg.coef_}')

# %%
resid = y-y_hat
plt.plot(x, resid)
np.max(np.abs(resid))

# %%
import statsmodels.api as sm

linear_residuals = resid
sm.graphics.tsa.plot_acf(linear_residuals, lags=20)
plt.show()
sm.graphics.tsa.plot_pacf(linear_residuals, lags=20)
plt.show()

# %%
from statsmodels.tsa.ar_model import AutoReg

xtrain = x
# linear_residuals
p = 2 # order
model = AutoReg(linear_residuals, lags=p).fit()
print(model.summary())

model_predictions = model.predict()
const = model.params[0]
phi1 = model.params[1]
phi2 = model.params[2]

# repeat down column to create column vector
model_predictions_calc = np.repeat([[np.nan]], len(xtrain), 0)
t_start = 2
model_predictions_calc[t_start:] = const + phi1 * linear_residuals[t_start-1:-1] + phi2 * linear_residuals[t_start-2:-2]

plt.plot(xtrain, linear_residuals, label='linear residuals data')
plt.plot(xtrain, model_predictions, 'r', label='fitted model')
plt.plot(xtrain, model_predictions_calc, 'kx', label='fitted model hand calc')
plt.legend()
plt.plot()
plt.title('AR model of linear residuals, training data')


# %%
# linear trend model params
alpha0 = reg.intercept_
alpha1 = reg.coef_[0,0]
# AR model params
const = model.params[0]
phi1 = model.params[1]
phi2 = model.params[2]

# generate predictions across full training data set to check model


xtrain = dftrain.monthIdx.values
ytrain = dftrain.CPI.values
ntrain = len(xtrain)

preds = np.repeat(np.nan, ntrain) # fill in first 2 with something since don't predict first 2

for i in range(0,ntrain):
    # skip first 2 (start at 2) since AR model needs access to previous 2
    if i < 2:
        continue
    t = xtrain[i]
    # big X(t-1) and X(t-2) for AR
    X_t_1 = linear_residuals[i-1,0]
    X_t_2 = linear_residuals[i-2,0]
    # X_t_1 = dftrain.CPI[ dftrain.monthIdx == (t-1) ].item()
    # X_t_2 = dftrain.CPI[ dftrain.monthIdx == (t-2) ].item()
    # model ( trend + AR )
    pred_t = alpha0 + alpha1 * t + phi1 * X_t_1 + phi2 * X_t_2 + const
    preds[i] = pred_t

preds_training = preds

plt.plot(xtrain, ytrain, label='original data training')
plt.plot(xtrain, preds_training, 'r', label='fitted line training')
plt.legend()
plt.title('model(trend + AR)')


# %%
# train model (with treand and AR) using AutoReg
det_comp = DeterministicProcess(xtrain, constant=True, order=1)
model = AutoReg(ytrain, deterministic=det_comp, lags=2).fit()
preds_training_ardp = model.predict()

plt.plot(xtrain, ytrain, label='original data training')
plt.plot(xtrain, preds_training_ardp, 'r', label='preds training')
plt.legend()
plt.title('model(trend + AR: using AutoReg/DetermProc)')
model.summary()


# start index of validation data
startIdx = dftrain.monthIdx.values[-1]
forecast = model.predict(start=startIdx, end=startIdx+2)
plt.plot(np.arange(startIdx,startIdx+3), forecast, 'k', label='forecast')
plt.legend()
plt.show()

const = model.params[0]
trend = model.params[1]
phi1 = model.params[2]
phi2 = model.params[3]

# verify model params fit training
preds_training_loop = np.repeat(np.nan, ntrain) 
for i in range(ntrain):
    if i < 2:
        continue
    t = xtrain[i]
    pred_t = const + trend * t + phi1 * ytrain[i-1] + phi2 * ytrain[t-2]
    preds_training_loop[i] = pred_t

plt.plot(xtrain, ytrain, label='original data training')
plt.plot(xtrain, preds_training_loop, 'r', label='preds training')
plt.legend()
plt.title('model(trend + AR: using AutoReg/DetermProc, loop)')

#%%
# 1 month forecasts for rest of data (after training data)
# 

# start index of validation data is after training
startIdx = dftrain.monthIdx.values[-1] + 1
endIdx = dfcpi.monthIdx.values[-1]
squ_err_arr = np.array([])
pred_t_all = np.array([])
x_all = np.array([])
y_all = np.array([])
for mi in range(startIdx, endIdx+1):
    # monthIdx same as df index  ** should have a better way
    y = dfcpi.CPI.values[mi]
    pred_t = const + trend * mi + phi1 * dfcpi.CPI.values[mi-1] + phi2 * dfcpi.CPI.values[mi-2]
    squ_err = (y - pred_t) ** 2
    squ_err_arr = np.append(squ_err_arr, squ_err)
    x_all = np.append(x_all, mi)
    y_all = np.append(y_all, y)
    pred_t_all = np.append(pred_t_all, pred_t)

rms = squ_err_arr.mean() ** .5
print(f' rms {rms}')

plt.plot(x_all, y_all, label="observations")
plt.plot(x_all,pred_t_all, label="forecasts")
plt.legend()
plt.titile('forecasts')


#%%

# percent Inflation Rate = (cpi_t - cpi_(t-1) )/ cpi_(t-1)

cpi_now = dfcpi.query('year == 2013 and month == 2').CPI.item()
cpi_prev = dfcpi.query('year == 2013 and month == 1').CPI.item()
IR = (cpi_now - cpi_prev) / cpi_prev * 100
print(f' percent IR {IR}')

# percent Inflation Rate-from-log = (ln(cpi_t) - ln(cpi_(t-1)))
IR_log = (np.log(cpi_now) - np.log(cpi_prev) ) * 100
print(f'percent IR from log {IR_log}')

# %%
