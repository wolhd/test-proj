
#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import datetime

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

y = linear_residuals
p = 2 # order
model = AutoReg(y, lags=p).fit()
print(model.summary())
# model_predictions = model.predict)
# plt.plot(x, y, label='raw y data')
# plt.plot(x, model_predictions, 'r', label='fitted model')
# plt.legend()
# plt.plot()


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

preds = ytrain[:2] # fill in first 2 with something since don't predict first 2

x_toIter = xtrain[2:] # skip first 2 (start at 2) since AR model needs access to previous 2
for t in x_toIter:
    # big X(t-1)  and X(t-2) for AR
    X_t_1 = dftrain.CPI[ dftrain.monthIdx == (t-1) ].item()
    X_t_2 = dftrain.CPI[ dftrain.monthIdx == (t-2) ].item()
    # model ( trend + AR )
    pred_t = alpha0 + alpha1 * t + phi1 * X_t_1 + phi2 * X_t_2 + const
    preds = np.append(preds, pred_t)

preds_training = preds

plt.plot(xtrain, ytrain, label='original data training')
plt.plot(xtrain, preds_training, 'r', label='fitted line training')
plt.legend()
plt.title('model(trend + AR)')


# %%
