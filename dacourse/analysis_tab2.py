

#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# %%
# skiprows is 0-indexed from start of file
df = pd.read_csv('CO2.csv', comment='"', header=0, skiprows=[55,56], skipinitialspace=True)
df = df.rename(columns={'Date': 'DateExcel', 'Date.1': 'Date'})
dforig = df
# define t = fraction years from jan.1958
t_i = (df.Yr-1958) + ((df.Mn-1 + .5) / 12)
df['t'] = t_i

#%%
# preprocess, remove -99.99
df = df[df.CO2 != -99.99]
df = df.reset_index(drop=True)

# %%
# split into train test
trainlen = round(len(df)*.8)
dftrain = df.iloc[:trainlen]
dftest = df.iloc[trainlen:]

print(f' train len {len(dftrain)}, test len {len(dftest)}' )

# %%
x = np.reshape(dftrain.t.values, [-1, 1])
y = np.reshape(dftrain.CO2.values, [-1, 1])
reg = LinearRegression().fit(x, y)
reg.score(x, y)
reg.intercept_
y_hat = reg.predict(x)
plt.plot(x, y, label='original data')
plt.plot(x, y_hat, 'r', label='fitted line')
plt.legend()
print(f'intercept {reg.intercept_} coef {reg.coef_}')

# %%
# residuals
linear_residuals = y - y_hat
plt.plot(x, linear_residuals,'o')

print(f'train rmse {mean_squared_error(y, y_hat)}')

# %%
# RMSE against TEST set

xtest = np.reshape(dftest.t.values, [-1,1])
ytest = np.reshape(dftest.CO2.values, [-1,1])
y_pred = reg.predict(xtest)


print(f'test rmse {mean_squared_error(ytest, y_pred) ** 0.5}')


print(f'test mape {mean_absolute_percentage_error(ytest, y_pred) * 100}')
plt.plot(xtest,ytest, label='test data')
plt.plot(xtest, y_pred, 'g', label='pred')

# %%
# fit quadratic
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
#  create x and x^2 col's
xx2 = poly.fit_transform(x)
xx2 = xx2[:,1:]  # remove 1st col since it is all 1's
reg = LinearRegression().fit(xx2, y)
y_hat = reg.predict(xx2)
plt.plot(x[:,0], y, label='original data')
plt.plot(x[:,0], y_hat, 'r', label='fitted line')
plt.legend()
print(f'intercept {reg.intercept_} coef {reg.coef_}')
regQuad = reg

# %%
quad_residuals = y - y_hat
plt.plot(x, quad_residuals,'o')
# %%
# RMSE against TEST set
# for quadratic

xtest = np.reshape(dftest.t.values, [-1,1])
xx2test = poly.fit_transform(xtest)
xx2test = xx2test[:,1:]  # remove 1st col since it is all 1's

ytest = np.reshape(dftest.CO2.values, [-1,1])
y_pred = reg.predict(xx2test)

print(f'test rmse {mean_squared_error(ytest, y_pred) ** 0.5}')


print(f'test mape {mean_absolute_percentage_error(ytest, y_pred) * 100}')
plt.plot(xtest,ytest, label='test data')
plt.plot(xtest, y_pred, 'g', label='pred')
plt.title('quadratic')

# %%
# fit cubic
poly = PolynomialFeatures(3)
#  create x  x^2  x^3 col's
xx3 = poly.fit_transform(x)
xx3 = xx3[:,1:]  # remove 1st col since it is all 1's
reg = LinearRegression().fit(xx3, y)
y_hat = reg.predict(xx3)
plt.plot(xx3[:,0], y, label='original data')
plt.plot(xx3[:,0], y_hat, 'r', label='fitted line')
plt.legend()
print(f'intercept {reg.intercept_} coef {reg.coef_}')


# %%
cubic_residuals = y - y_hat
plt.plot(x, cubic_residuals,'o')
# %%
# RMSE against TEST set
# for quadratic

xtest = np.reshape(dftest.t.values, [-1,1])
xx3test = poly.fit_transform(xtest)
xx3test = xx3test[:,1:]  # remove 1st col since it is all 1's

ytest = np.reshape(dftest.CO2.values, [-1,1])
y_pred = reg.predict(xx3test)

print(f'test rmse {mean_squared_error(ytest, y_pred) ** 0.5}')
print(f'test mape {mean_absolute_percentage_error(ytest, y_pred) * 100}')

plt.plot(xtest,ytest, label='test data')
plt.plot(xtest, y_pred, 'g', label='pred')
plt.title('cubic')

# %%
# periodic signal
y_pred_quad = regQuad.predict(xx2)
y_notrend = y - y_pred_quad
plt.plot(x, y_notrend)
dftrain['notrend_residuals'] = y_notrend
monthly_avg_notrend_resid = {} #monthly avg notrend residuals
for m in range(1,13):
    monthly_avg_notrend_resid[m] = dftrain[dftrain.Mn == m].notrend_residuals.mean()
monthly_avg_notrend_resid



# %%
