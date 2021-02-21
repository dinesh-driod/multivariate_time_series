import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from Autocorrelation import cal_auto_corr
from Pearson_Correlation_Coefficient import correlation_coefficent_cal
import warnings

warnings.filterwarnings("ignore")
register_matplotlib_converters()

'--------------------------Air Passengers Dataset-----------------------'
df = pd.read_csv('AirPassengers.csv', index_col='Month', parse_dates=True)
df.index.freq = 'MS'
y = df['#Passengers']
train, test = train_test_split(y, shuffle=False, test_size=0.2)
train.index.freq = 'MS'
test.index.freq = 'MS'
h = len(test)

print('************************** Air Passengers Dataset results ***********************')
#Average Method
def avg_method(train):
    y_hat_avg = np.mean(train)
    return y_hat_avg

train_pred_avg = []
for i in range(1,len(train)):
    res = avg_method(train.iloc[0:i])
    train_pred_avg.append(res)

test_forecast_avg1 = np.ones(len(test)) * avg_method(train)
test_forecast_avg = pd.DataFrame(test_forecast_avg1).set_index(test.index)
residual_error_avg = np.array(train[1:]) - np.array(train_pred_avg)
forecast_error_avg = test - test_forecast_avg1
MSE_train_avg = np.mean((residual_error_avg)**2)
MSE_test_avg = np.mean((forecast_error_avg)**2)
print('Mean Square Error of prediction errors for Average method: ', MSE_train_avg)
print('Mean Square Error of forecast errors for Average method: ', MSE_test_avg)
mean_pred_avg = np.mean(residual_error_avg)
var_pred_avg = np.var(residual_error_avg)
var_forecast_avg = np.var(forecast_error_avg)
print('Mean of prediction errors for Average method: ', mean_pred_avg)
print('Variance of prediction errors for Average method: ', var_pred_avg)
print('Variance of forecast errors for Average method: ', var_forecast_avg)

# Naive Method
def naive_method(t):
    return t

naive_train_pred = []
for i in range(0, len(train)-1):
    res = naive_method(train[i])
    naive_train_pred.append(res)

res = np.ones(len(test)) * train[-1]
naive_test_forecast1 = np.ones(len(test)) * res
naive_test_forecast = pd.DataFrame(naive_test_forecast1).set_index(test.index)
residual_error_naive = np.array(train[1:]) - np.array(naive_train_pred)
forecast_error_naive = test - naive_test_forecast1
MSE_train_naive = np.mean((residual_error_naive)**2)
MSE_test_naive = np.mean((forecast_error_naive)**2)
print('Mean Square Error of prediction errors for Naive method: ', MSE_train_naive)
print('Mean Square Error of forecast errors for Naive method: ', MSE_test_naive)
mean_pred_naive = np.mean(residual_error_naive)
var_pred_naive = np.var(residual_error_naive)
var_forecast_naive = np.var(forecast_error_naive)
print('Mean of prediction errors for Naive method: ', mean_pred_naive)
print('Variance of prediction errors for Naive method: ', var_pred_naive)
print('Variance of forecast errors for Naive method: ', var_forecast_naive)

# Drift method
def drift_method(t, h):
    y_hat_drift = t[len(t)-1] + h*((t[len(t)-1]-t[0])/(len(t) - 1))
    return y_hat_drift

drift_train_forecast=[]
for i in range(1, len(train)):
    if i == 1:
        drift_train_forecast.append(train[0])
    else:
        h = 1
        res = drift_method(train[0:i], h)
        drift_train_forecast.append(res)

drift_test_forecast1=[]
for h in range(1, len(test)+1):
    res = drift_method(train, h)
    drift_test_forecast1.append(res)

drift_test_forecast = pd.DataFrame(drift_test_forecast1).set_index(test.index)
residual_error_drift = np.array(train[1:]) - np.array(drift_train_forecast)
forecast_error_drift = np.array(test) - np.array(drift_test_forecast1)
MSE_train_drift = np.mean((residual_error_drift)**2)
MSE_test_drift = np.mean((forecast_error_drift)**2)
print('Mean Square Error of prediction errors for Drift method: ', MSE_train_drift)
print('Mean Square Error of forecast errors for Drift method: ', MSE_test_drift)
mean_pred_drift = np.mean(residual_error_drift)
var_pred_drift = np.var(residual_error_drift)
var_forecast_drift = np.var(forecast_error_drift)
print('Mean of prediction errors for Drift method: ', mean_pred_drift)
print('Variance of prediction errors for Drift method: ', var_pred_drift)
print('Variance of forecast errors for Drift method: ', var_forecast_drift)

#SES Method
def ses(t, damping_factor, l0):
    yhat4 = []
    yhat4.append(l0)
    for i in range(1, len(t)-1):
        res = damping_factor*(t[i]) + (1-damping_factor)*(yhat4[i-1])
        yhat4.append(res)
    return yhat4

l0 = train[0]
ses_train_pred = ses(train, 0.50, l0)
ses_test_forecast1 = np.ones(len(test)) * (0.5*(train[-1]) + (1-0.5)*(ses_train_pred[-1]))
ses_test_forecast = pd.DataFrame(ses_test_forecast1).set_index(test.index)
residual_error_ses = np.array(train[1:]) - np.array(ses_train_pred)
forecast_error_ses = np.array(test) - np.array(ses_test_forecast1)
MSE_train_SES = np.mean((residual_error_ses)**2)
MSE_test_SES = np.mean((forecast_error_ses)**2)
print('Mean Square Error of prediction errors for SES method: ', MSE_train_SES)
print('Mean Square Error of forecast errors for SES method: ', MSE_test_SES)
mean_pred_SES = np.mean(residual_error_ses)
var_pred_SES = np.var(residual_error_ses)
var_forecast_SES = np.var(forecast_error_ses)
print('Mean of prediction errors for SES method: ', mean_pred_SES)
print('Variance of prediction errors for SES method: ', var_pred_SES)
print('Variance of forecast errors for SES method: ', var_forecast_SES)

# SES Method using statsmodels for alpha=0.5
# ses_train = train.ewm(alpha=0.5, adjust=False).mean()  # Another way of doing it
ses_model1 = SimpleExpSmoothing(train)
ses_fitted_model1 = ses_model1.fit(smoothing_level=0.5, optimized=False)
ses_train_pred1 = ses_fitted_model1.fittedvalues.shift(-1)
ses_test_forecast1 = ses_fitted_model1.forecast(steps=len(test))
ses_test_forecast1 = pd.DataFrame(ses_test_forecast1).set_index(test.index)
MSE_test_SES1 = np.square(np.subtract(test.values, np.ndarray.flatten(ses_test_forecast1.values))).mean()

# Holt's Linear Trend
holtl_fitted_model = ets.ExponentialSmoothing(train, trend='multiplicative', damped=True, seasonal=None).fit()
holtl_train_pred = holtl_fitted_model.fittedvalues
holtl_test_forecast = holtl_fitted_model.forecast(steps=len(test))
holtl_test_forecast = pd.DataFrame(holtl_test_forecast).set_index(test.index)
residual_error_holtl = np.subtract(train.values, np.ndarray.flatten(holtl_train_pred.values))
forecast_error_holtl = np.subtract(test.values, np.ndarray.flatten(holtl_test_forecast.values))
MSE_train_holtl = np.mean((residual_error_holtl)**2)
MSE_test_holtl = np.mean((forecast_error_holtl)**2)
print("Mean Square Error of prediction errors for Holt's Linear method: ", MSE_train_holtl)
print("Mean Square Error of forecast errors for Holt's Linear method: ", MSE_test_holtl)
mean_pred_holtl = np.mean(residual_error_holtl)
var_pred_holtl = np.var(residual_error_holtl)
var_forecast_holtl = np.var(forecast_error_holtl)
print("Mean of prediction errors for Holt's Linear method: ", mean_pred_holtl)
print("Variance of prediction errors for Holt's Linear method: ", var_pred_holtl)
print("Variance of forecast errors for Holt's Linear method: ", var_forecast_holtl)

# Holt's Winter Seasonal Trend
holtw_fitted_model = ets.ExponentialSmoothing(train, trend='mul', damped=True, seasonal='mul', seasonal_periods=12).fit()
holtw_train_pred = holtw_fitted_model.fittedvalues
holtw_test_forecast = holtw_fitted_model.forecast(steps=len(test))
holtw_test_forecast = pd.DataFrame(holtw_test_forecast).set_index(test.index)

residual_error_holtw = np.subtract(train.values, np.ndarray.flatten(holtw_train_pred.values))
forecast_error_holtw = np.subtract(test.values, np.ndarray.flatten(holtw_test_forecast.values))
MSE_train_holtw = np.mean((residual_error_holtw)**2)
MSE_test_holtw = np.mean((forecast_error_holtw)**2)
print("Mean Square Error of prediction errors for Holt's Winter Seasonal method: ", MSE_train_holtw)
print("Mean Square Error of forecast errors for Holt's Winter Seasonal method: ", MSE_test_holtw)
mean_pred_holtw = np.mean(residual_error_holtw)
var_pred_holtw = np.var(residual_error_holtw)
var_forecast_holtw = np.var(forecast_error_holtw)
print("Mean of prediction errors for Holt's Winter Seasonal method: ", mean_pred_holtw)
print("Variance of prediction errors for Holt's Winter Seasonal method: ", var_pred_holtw)
print("Variance of forecast errors for Holt's Winter Seasonal method: ", var_forecast_holtw)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(test_forecast_avg, label='Average h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Passengers')
plt.title('Average Method - Air Passengers')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(naive_test_forecast, label='Naive h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Passengers')
plt.title('Naive Method - Air Passengers')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(drift_test_forecast, label='Drift h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Passengers')
plt.title('Drift Method - Air Passengers')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(ses_test_forecast, label='Simple Exponential Smoothing h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Passengers')
plt.title('SES Method - Air Passengers')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtl_test_forecast, label="Holt's Linear h-step prediction")
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Passengers')
plt.title("Holt's Linear Method - Air Passengers")
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtw_test_forecast, label="Holt's Winter Seasonal h-step prediction")
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Passengers')
plt.title("Holt's Winter Seasonal Method - Air Passengers")
plt.legend(loc='upper left')
plt.show()

# Auto_correlation for forecast errors and Q value for prediction errors and forecast errors
#Average Method
k = len(test)
lags = 15
avg_forecast_acf = cal_auto_corr(forecast_error_avg, lags)
Q_forecast_avg = k * np.sum(np.array(avg_forecast_acf[lags:])**2)
print('Q value of forecast errors for Average method: ', Q_forecast_avg)
plt.figure()
plt.stem(range(-(lags-1),lags), avg_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Average Method)')
plt.show()

# Naive method
k = len(test)
lags = 15
naive_forecast_acf = cal_auto_corr(forecast_error_naive, lags)
Q_forecast_naive = k * np.sum(np.array(naive_forecast_acf[lags:])**2)
print('Q value of forecast errors for Naive method: ', Q_forecast_naive)
plt.figure()
plt.stem(range(-(lags-1),lags), naive_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Naive Method)')
plt.show()

# Drift Method
k = len(test)
lags = 15
drift_forecast_acf = cal_auto_corr(forecast_error_drift, lags)
Q_forecast_drift = k * np.sum(np.array(drift_forecast_acf[lags:])**2)
print('Q value of forecast errors for Drift method: ', Q_forecast_drift)
plt.figure()
plt.stem(range(-(lags-1),lags), drift_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Drift Method)')
plt.show()

# SES method
k = len(test)
lags = 15
ses_forecast_acf = cal_auto_corr(forecast_error_ses, lags)
Q_forecast_SES = k * np.sum(np.array(ses_forecast_acf[lags:])**2)
print('Q value of forecast errors for SES method: ', Q_forecast_SES)
plt.figure()
plt.stem(range(-(lags-1), lags), ses_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (SES Method)')
plt.show()

# holt's linear method
k = len(test)
lags = 15
holtl_forecast_acf = cal_auto_corr(forecast_error_holtl, lags)
Q_forecast_holtl = k * np.sum(np.array(holtl_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Linear method: ", Q_forecast_holtl)
plt.figure()
plt.stem(range(-(lags-1), lags), holtl_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's Linear Method)")
plt.show()

# holt's Winter Seasonal method
k = len(test)
lags = 15
holtw_forecast_acf = cal_auto_corr(forecast_error_holtw, lags)
Q_forecast_holtw = k * np.sum(np.array(holtw_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Winter Seasonal method: ", Q_forecast_holtw)
plt.figure()
plt.stem(range(-(lags-1), lags), holtw_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's winter Seasonal Method)")
plt.show()

corr_avg = correlation_coefficent_cal(forecast_error_avg, test)
corr_naive = correlation_coefficent_cal(forecast_error_naive, test)
corr_drift = correlation_coefficent_cal(forecast_error_drift, test)
corr_ses = correlation_coefficent_cal(forecast_error_ses, test)
corr_holtl = correlation_coefficent_cal(forecast_error_holtl, test)
corr_holtw = correlation_coefficent_cal(forecast_error_holtw, test)
print("Correlation Coefficient between Forecast Error and Test set for Average Method: {}".format(corr_avg))
print("Correlation Coefficient between Forecast Error and Test set for Naive Method: {}".format(corr_naive))
print("Correlation Coefficient between Forecast Error and Test set for Drift Method: {}".format(corr_drift))
print("Correlation Coefficient between Forecast Error and Test set for SES Method: {}".format(corr_ses))
print("Correlation Coefficient between Forecast Error and Test set for Holt's Linear Method: {}".format(corr_holtl))
print("Correlation Coefficient between Forecast Error and Test set for Holt's winter seasonal Method: {}".format(corr_holtw))
d = {'Methods':['Average', 'Naive', 'Drift', 'SES', "HoltL", "HoltW"],
     'Q_val': [round(Q_forecast_avg, 2), round(Q_forecast_naive, 2), round(Q_forecast_drift,2), round(Q_forecast_SES,2), round(Q_forecast_holtl,2), round(Q_forecast_holtw,2)],
     'MSE(P)': [round(MSE_train_avg,2), round(MSE_train_naive,2), round(MSE_train_drift,2), round(MSE_train_SES,2), round(MSE_train_holtl,2), round(MSE_train_holtw,2)],
     'MSE(F)': [round(MSE_test_avg,2), round(MSE_test_naive,2), round(MSE_test_drift,2), round(MSE_test_SES,2), round(MSE_test_holtl,2), round(MSE_test_holtw,2)],
     'var(P)': [round(var_pred_avg,2), round(var_pred_naive,2), round(var_pred_drift,2), round(var_pred_SES,2), round(var_pred_holtl,2), round(var_pred_holtw,2)],
     'var(F)':[round(var_forecast_avg,2), round(var_forecast_naive,2), round(var_forecast_drift,2), round(var_forecast_SES,2), round(var_forecast_holtl,2), round(var_forecast_holtw,2)],
     'corrcoeff':[round(corr_avg,2), round(corr_naive,2), round(corr_drift,2), round(corr_ses,2), round(corr_holtl,2), round(corr_holtw,2)]}
df = pd.DataFrame(data=d)
df = df.set_index('Methods')
pd.set_option('display.max_columns', None)
print(df)


'-----------------------------------------------------shampoo Dataset--------------------------------------------------'
df = pd.read_csv('shampoo.csv', index_col='Month', parse_dates=True)
df.index.freq = 'MS'
y = df['Sales']
y.index = pd.date_range(start='2001-01-01', end='2003-12-01', freq='MS')
train, test = train_test_split(y, shuffle=False, test_size=0.2)
train.index.freq = 'MS'
test.index.freq = 'MS'
h = len(test)
print('************************* Shampoo Dataset results ***********************')
#Average Method
def avg_method(train):
    y_hat_avg = np.mean(train)
    return y_hat_avg

train_pred_avg = []
for i in range(1,len(train)):
    res = avg_method(train.iloc[0:i])
    train_pred_avg.append(res)

test_forecast_avg1 = np.ones(len(test)) * avg_method(train)
test_forecast_avg = pd.DataFrame(test_forecast_avg1).set_index(test.index)
residual_error_avg = np.array(train[1:]) - np.array(train_pred_avg)
forecast_error_avg = test - test_forecast_avg1
MSE_train_avg = np.mean((residual_error_avg)**2)
MSE_test_avg = np.mean((forecast_error_avg)**2)
print('Mean Square Error of prediction errors for Average method: ', MSE_train_avg)
print('Mean Square Error of forecast errors for Average method: ', MSE_test_avg)
mean_pred_avg = np.mean(residual_error_avg)
var_pred_avg = np.var(residual_error_avg)
var_forecast_avg = np.var(forecast_error_avg)
print('Mean of prediction errors for Average method: ', mean_pred_avg)
print('Variance of prediction errors for Average method: ', var_pred_avg)
print('Variance of forecast errors for Average method: ', var_forecast_avg)

# Naive Method
def naive_method(t):
    return t

naive_train_pred = []
for i in range(0, len(train)-1):
    res = naive_method(train[i])
    naive_train_pred.append(res)

res = np.ones(len(test)) * train[-1]
naive_test_forecast1 = np.ones(len(test)) * res
naive_test_forecast = pd.DataFrame(naive_test_forecast1).set_index(test.index)
residual_error_naive = np.array(train[1:]) - np.array(naive_train_pred)
forecast_error_naive = test - naive_test_forecast1
MSE_train_naive = np.mean((residual_error_naive)**2)
MSE_test_naive = np.mean((forecast_error_naive)**2)
print('Mean Square Error of prediction errors for Naive method: ', MSE_train_naive)
print('Mean Square Error of forecast errors for Naive method: ', MSE_test_naive)
mean_pred_naive = np.mean(residual_error_naive)
var_pred_naive = np.var(residual_error_naive)
var_forecast_naive = np.var(forecast_error_naive)
print('Mean of prediction errors for Naive method: ', mean_pred_naive)
print('Variance of prediction errors for Naive method: ', var_pred_naive)
print('Variance of forecast errors for Naive method: ', var_forecast_naive)

# Drift method
def drift_method(t, h):
    y_hat_drift = t[len(t)-1] + h*((t[len(t)-1]-t[0])/(len(t) - 1))
    return y_hat_drift

drift_train_forecast=[]
for i in range(1, len(train)):
    if i == 1:
        drift_train_forecast.append(train[0])
    else:
        h = 1
        res = drift_method(train[0:i], h)
        drift_train_forecast.append(res)

drift_test_forecast1=[]
for h in range(1, len(test)+1):
    res = drift_method(train, h)
    drift_test_forecast1.append(res)

drift_test_forecast = pd.DataFrame(drift_test_forecast1).set_index(test.index)
residual_error_drift = np.array(train[1:]) - np.array(drift_train_forecast)
forecast_error_drift = np.array(test) - np.array(drift_test_forecast1)
MSE_train_drift = np.mean((residual_error_drift)**2)
MSE_test_drift = np.mean((forecast_error_drift)**2)
print('Mean Square Error of prediction errors for Drift method: ', MSE_train_drift)
print('Mean Square Error of forecast errors for Drift method: ', MSE_test_drift)
mean_pred_drift = np.mean(residual_error_drift)
var_pred_drift = np.var(residual_error_drift)
var_forecast_drift = np.var(forecast_error_drift)
print('Mean of prediction errors for Drift method: ', mean_pred_drift)
print('Variance of prediction errors for Drift method: ', var_pred_drift)
print('Variance of forecast errors for Drift method: ', var_forecast_drift)

#SES Method
def ses(t, damping_factor, l0):
    yhat4 = []
    yhat4.append(l0)
    for i in range(1, len(t)-1):
        res = damping_factor*(t[i]) + (1-damping_factor)*(yhat4[i-1])
        yhat4.append(res)
    return yhat4

l0 = train[0]
ses_train_pred = ses(train, 0.50, l0)
ses_test_forecast1 = np.ones(len(test)) * (0.5*(train[-1]) + (1-0.5)*(ses_train_pred[-1]))
ses_test_forecast = pd.DataFrame(ses_test_forecast1).set_index(test.index)
residual_error_ses = np.array(train[1:]) - np.array(ses_train_pred)
forecast_error_ses = np.array(test) - np.array(ses_test_forecast1)
MSE_train_SES = np.mean((residual_error_ses)**2)
MSE_test_SES = np.mean((forecast_error_ses)**2)
print('Mean Square Error of prediction errors for SES method: ', MSE_train_SES)
print('Mean Square Error of forecast errors for SES method: ', MSE_test_SES)
mean_pred_SES = np.mean(residual_error_ses)
var_pred_SES = np.var(residual_error_ses)
var_forecast_SES = np.var(forecast_error_ses)
print('Mean of prediction errors for SES method: ', mean_pred_SES)
print('Variance of prediction errors for SES method: ', var_pred_SES)
print('Variance of forecast errors for SES method: ', var_forecast_SES)

# SES Method using statsmodels for alpha=0.5
# ses_train = train.ewm(alpha=0.5, adjust=False).mean()  # Another way of doing it
ses_model1 = SimpleExpSmoothing(train)
ses_fitted_model1 = ses_model1.fit(smoothing_level=0.5, optimized=False)
ses_train_pred1 = ses_fitted_model1.fittedvalues.shift(-1)
ses_test_forecast1 = ses_fitted_model1.forecast(steps=len(test))
ses_test_forecast1 = pd.DataFrame(ses_test_forecast1).set_index(test.index)
MSE_test_SES1 = np.square(np.subtract(test.values, np.ndarray.flatten(ses_test_forecast1.values))).mean()

# Holt's Linear Trend
holtl_fitted_model = ets.ExponentialSmoothing(train, trend='multiplicative', damped=True, seasonal=None).fit()
holtl_train_pred = holtl_fitted_model.fittedvalues
holtl_test_forecast = holtl_fitted_model.forecast(steps=len(test))
holtl_test_forecast = pd.DataFrame(holtl_test_forecast).set_index(test.index)
residual_error_holtl = np.subtract(train.values, np.ndarray.flatten(holtl_train_pred.values))
forecast_error_holtl = np.subtract(test.values, np.ndarray.flatten(holtl_test_forecast.values))
MSE_train_holtl = np.mean((residual_error_holtl)**2)
MSE_test_holtl = np.mean((forecast_error_holtl)**2)
print("Mean Square Error of prediction errors for Holt's Linear method: ", MSE_train_holtl)
print("Mean Square Error of forecast errors for Holt's Linear method: ", MSE_test_holtl)
mean_pred_holtl = np.mean(residual_error_holtl)
var_pred_holtl = np.var(residual_error_holtl)
var_forecast_holtl = np.var(forecast_error_holtl)
print("Mean of prediction errors for Holt's Linear method: ", mean_pred_holtl)
print("Variance of prediction errors for Holt's Linear method: ", var_pred_holtl)
print("Variance of forecast errors for Holt's Linear method: ", var_forecast_holtl)

# Holt's Winter Seasonal Trend
holtw_fitted_model = ets.ExponentialSmoothing(train, trend='mul', damped=True, seasonal='mul', seasonal_periods=12).fit()
holtw_train_pred = holtw_fitted_model.fittedvalues
holtw_test_forecast = holtw_fitted_model.forecast(steps=len(test))
holtw_test_forecast = pd.DataFrame(holtw_test_forecast).set_index(test.index)
residual_error_holtw = np.subtract(train.values, np.ndarray.flatten(holtw_train_pred.values))
forecast_error_holtw = np.subtract(test.values, np.ndarray.flatten(holtw_test_forecast.values))
MSE_train_holtw = np.mean((residual_error_holtw)**2)
MSE_test_holtw = np.mean((forecast_error_holtw)**2)
print("Mean Square Error of prediction errors for Holt's Winter Seasonal method: ", MSE_train_holtw)
print("Mean Square Error of forecast errors for Holt's Winter Seasonal method: ", MSE_test_holtw)
mean_pred_holtw = np.mean(residual_error_holtw)
var_pred_holtw = np.var(residual_error_holtw)
var_forecast_holtw = np.var(forecast_error_holtw)
print("Mean of prediction errors for Holt's Winter Seasonal method: ", mean_pred_holtw)
print("Variance of prediction errors for Holt's Winter Seasonal method: ", var_pred_holtw)
print("Variance of forecast errors for Holt's Winter Seasonal method: ", var_forecast_holtw)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(test_forecast_avg, label='Average h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Sales')
plt.title('Average Method - Shampoo')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(naive_test_forecast, label='Naive h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Sales')
plt.title('Naive Method - Shampoo')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(drift_test_forecast, label='Drift h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Sales')
plt.title('Drift Method - Shampoo')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(ses_test_forecast, label='Simple Exponential Smoothing h-step prediction')
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Sales')
plt.title('SES Method - Shampoo')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtl_test_forecast, label="Holt's Linear h-step prediction")
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Sales')
plt.title("Holt's Linear Method - Shampoo")
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtw_test_forecast, label="Holt's Winter Seasonal h-step prediction")
plt.xlabel('Time (Monthly)')
plt.ylabel('Number of Sales')
plt.title("Holt's Winter Seasonal Method - Shampoo")
plt.legend(loc='upper left')
plt.show()

# Auto_correlation for forecast errors and Q value for prediction errors and forecast errors
#Average Method
k = len(test)
lags = 15
avg_forecast_acf = cal_auto_corr(forecast_error_avg, lags)
Q_forecast_avg = k * np.sum(np.array(avg_forecast_acf[lags:])**2)
print('Q value of forecast errors for Average method: ', Q_forecast_avg)
plt.figure()
plt.stem(range(-(lags-1),lags), avg_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Average Method)')
plt.show()

# Naive method
k = len(test)
lags = 15
naive_forecast_acf = cal_auto_corr(forecast_error_naive, lags)
Q_forecast_naive = k * np.sum(np.array(naive_forecast_acf[lags:])**2)
print('Q value of forecast errors for Naive method: ', Q_forecast_naive)
plt.figure()
plt.stem(range(-(lags-1),lags), naive_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Naive Method)')
plt.show()

# Drift Method
k = len(test)
lags = 15
drift_forecast_acf = cal_auto_corr(forecast_error_drift, lags)
Q_forecast_drift = k * np.sum(np.array(drift_forecast_acf[lags:])**2)
print('Q value of forecast errors for Drift method: ', Q_forecast_drift)
plt.figure()
plt.stem(range(-(lags-1),lags), drift_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Drift Method)')
plt.show()

# SES method
k = len(test)
lags = 15
ses_forecast_acf = cal_auto_corr(forecast_error_ses, lags)
Q_forecast_SES = k * np.sum(np.array(ses_forecast_acf[lags:])**2)
print('Q value of forecast errors for SES method: ', Q_forecast_SES)
plt.figure()
plt.stem(range(-(lags-1), lags), ses_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (SES Method)')
plt.show()

# holt's linear method
k = len(test)
lags = 15
holtl_forecast_acf = cal_auto_corr(forecast_error_holtl, lags)
Q_forecast_holtl = k * np.sum(np.array(holtl_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Linear method: ", Q_forecast_holtl)
plt.figure()
plt.stem(range(-(lags-1), lags), holtl_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's Linear Method)")
plt.show()

# holt's Winter Seasonal method
k = len(test)
lags = 15
holtw_forecast_acf = cal_auto_corr(forecast_error_holtw, lags)
Q_forecast_holtw = k * np.sum(np.array(holtw_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Winter Seasonal method: ", Q_forecast_holtw)
plt.figure()
plt.stem(range(-(lags-1), lags), holtw_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's winter Seasonal Method)")
plt.show()

corr_avg = correlation_coefficent_cal(forecast_error_avg, test)
corr_naive = correlation_coefficent_cal(forecast_error_naive, test)
corr_drift = correlation_coefficent_cal(forecast_error_drift, test)
corr_ses = correlation_coefficent_cal(forecast_error_ses, test)
corr_holtl = correlation_coefficent_cal(forecast_error_holtl, test)
corr_holtw = correlation_coefficent_cal(forecast_error_holtw, test)
print("Correlation Coefficient between Forecast Error and Test set for Average Method: {}".format(corr_avg))
print("Correlation Coefficient between Forecast Error and Test set for Naive Method: {}".format(corr_naive))
print("Correlation Coefficient between Forecast Error and Test set for Drift Method: {}".format(corr_drift))
print("Correlation Coefficient between Forecast Error and Test set for SES Method: {}".format(corr_ses))
print("Correlation Coefficient between Forecast Error and Test set for Holt's Linear Method: {}".format(corr_holtl))
print("Correlation Coefficient between Forecast Error and Test set for Holt's winter seasonal Method: {}".format(corr_holtw))
d = {'Methods':['Average', 'Naive', 'Drift', 'SES', "HoltL", "HoltW"],
     'Q_val': [round(Q_forecast_avg, 2), round(Q_forecast_naive, 2), round(Q_forecast_drift,2), round(Q_forecast_SES,2), round(Q_forecast_holtl,2), round(Q_forecast_holtw,2)],
     'MSE(P)': [round(MSE_train_avg,2), round(MSE_train_naive,2), round(MSE_train_drift,2), round(MSE_train_SES,2), round(MSE_train_holtl,2), round(MSE_train_holtw,2)],
     'MSE(F)': [round(MSE_test_avg,2), round(MSE_test_naive,2), round(MSE_test_drift,2), round(MSE_test_SES,2), round(MSE_test_holtl,2), round(MSE_test_holtw,2)],
     'var(P)': [round(var_pred_avg,2), round(var_pred_naive,2), round(var_pred_drift,2), round(var_pred_SES,2), round(var_pred_holtl,2), round(var_pred_holtw,2)],
     'var(F)':[round(var_forecast_avg,2), round(var_forecast_naive,2), round(var_forecast_drift,2), round(var_forecast_SES,2), round(var_forecast_holtl,2), round(var_forecast_holtw,2)],
     'corrcoeff':[round(corr_avg,2), round(corr_naive,2), round(corr_drift,2), round(corr_ses,2), round(corr_holtl,2), round(corr_holtw,2)]}
df = pd.DataFrame(data=d)
df = df.set_index('Methods')
pd.set_option('display.max_columns', None)
print(df)


'---------------------------------------------- daily-total-female-births Dataset -------------------------------------'
df = pd.read_csv('daily-total-female-births.csv', index_col='Date', parse_dates=True)
df.index.freq = 'D'
y = df['Births']
y.index = pd.date_range(start='1959-01-01', end='1959-12-31', freq='D')
train, test = train_test_split(y, shuffle=False, test_size=0.2)
train.index.freq = 'D'
test.index.freq = 'D'
h = len(test)
print('************************** daily-total-female-births Dataset results ***********************')
#Average Method
def avg_method(train):
    y_hat_avg = np.mean(train)
    return y_hat_avg

train_pred_avg = []
for i in range(1,len(train)):
    res = avg_method(train.iloc[0:i])
    train_pred_avg.append(res)

test_forecast_avg1 = np.ones(len(test)) * avg_method(train)
test_forecast_avg = pd.DataFrame(test_forecast_avg1).set_index(test.index)
residual_error_avg = np.array(train[1:]) - np.array(train_pred_avg)
forecast_error_avg = test - test_forecast_avg1
MSE_train_avg = np.mean((residual_error_avg)**2)
MSE_test_avg = np.mean((forecast_error_avg)**2)
print('Mean Square Error of prediction errors for Average method: ', MSE_train_avg)
print('Mean Square Error of forecast errors for Average method: ', MSE_test_avg)
mean_pred_avg = np.mean(residual_error_avg)
var_pred_avg = np.var(residual_error_avg)
var_forecast_avg = np.var(forecast_error_avg)
print('Mean of prediction errors for Average method: ', mean_pred_avg)
print('Variance of prediction errors for Average method: ', var_pred_avg)
print('Variance of forecast errors for Average method: ', var_forecast_avg)

# Naive Method
def naive_method(t):
    return t

naive_train_pred = []
for i in range(0, len(train)-1):
    res = naive_method(train[i])
    naive_train_pred.append(res)

res = np.ones(len(test)) * train[-1]
naive_test_forecast1 = np.ones(len(test)) * res
naive_test_forecast = pd.DataFrame(naive_test_forecast1).set_index(test.index)
residual_error_naive = np.array(train[1:]) - np.array(naive_train_pred)
forecast_error_naive = test - naive_test_forecast1
MSE_train_naive = np.mean((residual_error_naive)**2)
MSE_test_naive = np.mean((forecast_error_naive)**2)
print('Mean Square Error of prediction errors for Naive method: ', MSE_train_naive)
print('Mean Square Error of forecast errors for Naive method: ', MSE_test_naive)
mean_pred_naive = np.mean(residual_error_naive)
var_pred_naive = np.var(residual_error_naive)
var_forecast_naive = np.var(forecast_error_naive)
print('Mean of prediction errors for Naive method: ', mean_pred_naive)
print('Variance of prediction errors for Naive method: ', var_pred_naive)
print('Variance of forecast errors for Naive method: ', var_forecast_naive)

# Drift method
def drift_method(t, h):
    y_hat_drift = t[len(t)-1] + h*((t[len(t)-1]-t[0])/(len(t) - 1))
    return y_hat_drift

drift_train_forecast=[]
for i in range(1, len(train)):
    if i == 1:
        drift_train_forecast.append(train[0])
    else:
        h = 1
        res = drift_method(train[0:i], h)
        drift_train_forecast.append(res)

drift_test_forecast1=[]
for h in range(1, len(test)+1):
    res = drift_method(train, h)
    drift_test_forecast1.append(res)

drift_test_forecast = pd.DataFrame(drift_test_forecast1).set_index(test.index)
residual_error_drift = np.array(train[1:]) - np.array(drift_train_forecast)
forecast_error_drift = np.array(test) - np.array(drift_test_forecast1)
MSE_train_drift = np.mean((residual_error_drift)**2)
MSE_test_drift = np.mean((forecast_error_drift)**2)
print('Mean Square Error of prediction errors for Drift method: ', MSE_train_drift)
print('Mean Square Error of forecast errors for Drift method: ', MSE_test_drift)
mean_pred_drift = np.mean(residual_error_drift)
var_pred_drift = np.var(residual_error_drift)
var_forecast_drift = np.var(forecast_error_drift)
print('Mean of prediction errors for Drift method: ', mean_pred_drift)
print('Variance of prediction errors for Drift method: ', var_pred_drift)
print('Variance of forecast errors for Drift method: ', var_forecast_drift)

#SES Method
def ses(t, damping_factor, l0):
    yhat4 = []
    yhat4.append(l0)
    for i in range(1, len(t)-1):
        res = damping_factor*(t[i]) + (1-damping_factor)*(yhat4[i-1])
        yhat4.append(res)
    return yhat4

l0 = train[0]
ses_train_pred = ses(train, 0.50, l0)
ses_test_forecast1 = np.ones(len(test)) * (0.5*(train[-1]) + (1-0.5)*(ses_train_pred[-1]))
ses_test_forecast = pd.DataFrame(ses_test_forecast1).set_index(test.index)
residual_error_ses = np.array(train[1:]) - np.array(ses_train_pred)
forecast_error_ses = np.array(test) - np.array(ses_test_forecast1)
MSE_train_SES = np.mean((residual_error_ses)**2)
MSE_test_SES = np.mean((forecast_error_ses)**2)
print('Mean Square Error of prediction errors for SES method: ', MSE_train_SES)
print('Mean Square Error of forecast errors for SES method: ', MSE_test_SES)
mean_pred_SES = np.mean(residual_error_ses)
var_pred_SES = np.var(residual_error_ses)
var_forecast_SES = np.var(forecast_error_ses)
print('Mean of prediction errors for SES method: ', mean_pred_SES)
print('Variance of prediction errors for SES method: ', var_pred_SES)
print('Variance of forecast errors for SES method: ', var_forecast_SES)

# SES Method using statsmodels for alpha=0.5
# ses_train = train.ewm(alpha=0.5, adjust=False).mean()  # Another way of doing it
ses_model1 = SimpleExpSmoothing(train)
ses_fitted_model1 = ses_model1.fit(smoothing_level=0.5, optimized=False)
ses_train_pred1 = ses_fitted_model1.fittedvalues.shift(-1)
ses_test_forecast1 = ses_fitted_model1.forecast(steps=len(test))
ses_test_forecast1 = pd.DataFrame(ses_test_forecast1).set_index(test.index)
MSE_test_SES1 = np.square(np.subtract(test.values, np.ndarray.flatten(ses_test_forecast1.values))).mean()

# Holt's Linear Trend
holtl_fitted_model = ets.ExponentialSmoothing(train, trend='add', damped=True, seasonal=None).fit()
holtl_train_pred = holtl_fitted_model.fittedvalues
holtl_test_forecast = holtl_fitted_model.forecast(steps=len(test))
holtl_test_forecast = pd.DataFrame(holtl_test_forecast).set_index(test.index)
residual_error_holtl = np.subtract(train.values, np.ndarray.flatten(holtl_train_pred.values))
forecast_error_holtl = np.subtract(test.values, np.ndarray.flatten(holtl_test_forecast.values))
MSE_train_holtl = np.mean((residual_error_holtl)**2)
MSE_test_holtl = np.mean((forecast_error_holtl)**2)
print("Mean Square Error of prediction errors for Holt's Linear method: ", MSE_train_holtl)
print("Mean Square Error of forecast errors for Holt's Linear method: ", MSE_test_holtl)
mean_pred_holtl = np.mean(residual_error_holtl)
var_pred_holtl = np.var(residual_error_holtl)
var_forecast_holtl = np.var(forecast_error_holtl)
print("Mean of prediction errors for Holt's Linear method: ", mean_pred_holtl)
print("Variance of prediction errors for Holt's Linear method: ", var_pred_holtl)
print("Variance of forecast errors for Holt's Linear method: ", var_forecast_holtl)

# Holt's Winter Seasonal Trend
holtw_fitted_model = ets.ExponentialSmoothing(train, trend='add', damped=True, seasonal='add').fit()
holtw_train_pred = holtw_fitted_model.fittedvalues
holtw_test_forecast = holtw_fitted_model.forecast(steps=len(test))
holtw_test_forecast = pd.DataFrame(holtw_test_forecast).set_index(test.index)
residual_error_holtw = np.subtract(train.values, np.ndarray.flatten(holtw_train_pred.values))
forecast_error_holtw = np.subtract(test.values, np.ndarray.flatten(holtw_test_forecast.values))
MSE_train_holtw = np.mean((residual_error_holtw)**2)
MSE_test_holtw = np.mean((forecast_error_holtw)**2)
print("Mean Square Error of prediction errors for Holt's Winter Seasonal method: ", MSE_train_holtw)
print("Mean Square Error of forecast errors for Holt's Winter Seasonal method: ", MSE_test_holtw)
mean_pred_holtw = np.mean(residual_error_holtw)
var_pred_holtw = np.var(residual_error_holtw)
var_forecast_holtw = np.var(forecast_error_holtw)
print("Mean of prediction errors for Holt's Winter Seasonal method: ", mean_pred_holtw)
print("Variance of prediction errors for Holt's Winter Seasonal method: ", var_pred_holtw)
print("Variance of forecast errors for Holt's Winter Seasonal method: ", var_forecast_holtw)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(test_forecast_avg, label='Average h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Number of Births')
plt.title('Average Method - daily-total-female-births')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(naive_test_forecast, label='Naive h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Number of Births')
plt.title('Naive Method - daily-total-female-births')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(drift_test_forecast, label='Drift h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Number of Births')
plt.title('Drift Method - daily-total-female-births')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(ses_test_forecast, label='Simple Exponential Smoothing h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Number of Births')
plt.title('SES Method - daily-total-female-births')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtl_test_forecast, label="Holt's Linear h-step prediction")
plt.xlabel('Time (Daily)')
plt.ylabel('Number of Births')
plt.title("Holt's Linear Method - daily-total-female-births")
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtw_test_forecast, label="Holt's Winter Seasonal h-step prediction")
plt.xlabel('Time (Daily)')
plt.ylabel('Number of Births')
plt.title("Holt's Winter Seasonal Method - daily-total-female-births")
plt.legend(loc='upper left')
plt.show()

# Auto_correlation for forecast errors and Q value for prediction errors and forecast errors
#Average Method
k = len(test)
lags = 15
avg_forecast_acf = cal_auto_corr(forecast_error_avg, lags)
Q_forecast_avg = k * np.sum(np.array(avg_forecast_acf[lags:])**2)
print('Q value of forecast errors for Average method: ', Q_forecast_avg)
plt.figure()
plt.stem(range(-(lags-1),lags), avg_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Average Method)')
plt.show()

# Naive method
k = len(test)
lags = 15
naive_forecast_acf = cal_auto_corr(forecast_error_naive, lags)
Q_forecast_naive = k * np.sum(np.array(naive_forecast_acf[lags:])**2)
print('Q value of forecast errors for Naive method: ', Q_forecast_naive)
plt.figure()
plt.stem(range(-(lags-1),lags), naive_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Naive Method)')
plt.show()

# Drift Method
k = len(test)
lags = 15
drift_forecast_acf = cal_auto_corr(forecast_error_drift, lags)
Q_forecast_drift = k * np.sum(np.array(drift_forecast_acf[lags:])**2)
print('Q value of forecast errors for Drift method: ', Q_forecast_drift)
plt.figure()
plt.stem(range(-(lags-1),lags), drift_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Drift Method)')
plt.show()

# SES method
k = len(test)
lags = 15
ses_forecast_acf = cal_auto_corr(forecast_error_ses, lags)
Q_forecast_SES = k * np.sum(np.array(ses_forecast_acf[lags:])**2)
print('Q value of forecast errors for SES method: ', Q_forecast_SES)
plt.figure()
plt.stem(range(-(lags-1), lags), ses_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (SES Method)')
plt.show()

# holt's linear method
k = len(test)
lags = 15
holtl_forecast_acf = cal_auto_corr(forecast_error_holtl, lags)
Q_forecast_holtl = k * np.sum(np.array(holtl_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Linear method: ", Q_forecast_holtl)
plt.figure()
plt.stem(range(-(lags-1), lags), holtl_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's Linear Method)")
plt.show()

# holt's Winter Seasonal method
k = len(test)
lags = 15
holtw_forecast_acf = cal_auto_corr(forecast_error_holtw, lags)
Q_forecast_holtw = k * np.sum(np.array(holtw_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Winter Seasonal method: ", Q_forecast_holtw)
plt.figure()
plt.stem(range(-(lags-1), lags), holtw_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's winter Seasonal Method)")
plt.show()

corr_avg = correlation_coefficent_cal(forecast_error_avg, test)
corr_naive = correlation_coefficent_cal(forecast_error_naive, test)
corr_drift = correlation_coefficent_cal(forecast_error_drift, test)
corr_ses = correlation_coefficent_cal(forecast_error_ses, test)
corr_holtl = correlation_coefficent_cal(forecast_error_holtl, test)
corr_holtw = correlation_coefficent_cal(forecast_error_holtw, test)
print("Correlation Coefficient between Forecast Error and Test set for Average Method: {}".format(corr_avg))
print("Correlation Coefficient between Forecast Error and Test set for Naive Method: {}".format(corr_naive))
print("Correlation Coefficient between Forecast Error and Test set for Drift Method: {}".format(corr_drift))
print("Correlation Coefficient between Forecast Error and Test set for SES Method: {}".format(corr_ses))
print("Correlation Coefficient between Forecast Error and Test set for Holt's Linear Method: {}".format(corr_holtl))
print("Correlation Coefficient between Forecast Error and Test set for Holt's winter seasonal Method: {}".format(corr_holtw))
d = {'Methods':['Average', 'Naive', 'Drift', 'SES', "HoltL", "HoltW"],
     'Q_val': [round(Q_forecast_avg, 2), round(Q_forecast_naive, 2), round(Q_forecast_drift,2), round(Q_forecast_SES,2), round(Q_forecast_holtl,2), round(Q_forecast_holtw,2)],
     'MSE(P)': [round(MSE_train_avg,2), round(MSE_train_naive,2), round(MSE_train_drift,2), round(MSE_train_SES,2), round(MSE_train_holtl,2), round(MSE_train_holtw,2)],
     'MSE(F)': [round(MSE_test_avg,2), round(MSE_test_naive,2), round(MSE_test_drift,2), round(MSE_test_SES,2), round(MSE_test_holtl,2), round(MSE_test_holtw,2)],
     'var(P)': [round(var_pred_avg,2), round(var_pred_naive,2), round(var_pred_drift,2), round(var_pred_SES,2), round(var_pred_holtl,2), round(var_pred_holtw,2)],
     'var(F)':[round(var_forecast_avg,2), round(var_forecast_naive,2), round(var_forecast_drift,2), round(var_forecast_SES,2), round(var_forecast_holtl,2), round(var_forecast_holtw,2)],
     'corrcoeff':[round(corr_avg,2), round(corr_naive,2), round(corr_drift,2), round(corr_ses,2), round(corr_holtl,2), round(corr_holtw,2)]}
df = pd.DataFrame(data=d)
df = df.set_index('Methods')
pd.set_option('display.max_columns', None)
print(df)


'--------------------------------------------------- Tute1 Dataset ----------------------------------------------------'
df = pd.read_csv('tute1.csv', header=0)
df['Date'] = pd.date_range(start='1981-3-1', end='2006-3-1', freq='Q')
y = df['Sales']
y.index = df.Date
train, test = train_test_split(y, shuffle=False, test_size=0.2)
train.index.freq = 'Q'
test.index.freq = 'Q'
h = len(test)

print('************************** Tute1 Dataset results ***********************')
#Average Method
def avg_method(train):
    y_hat_avg = np.mean(train)
    return y_hat_avg

train_pred_avg = []
for i in range(1,len(train)):
    res = avg_method(train.iloc[0:i])
    train_pred_avg.append(res)

test_forecast_avg1 = np.ones(len(test)) * avg_method(train)
test_forecast_avg = pd.DataFrame(test_forecast_avg1).set_index(test.index)
residual_error_avg = np.array(train[1:]) - np.array(train_pred_avg)
forecast_error_avg = test - test_forecast_avg1
MSE_train_avg = np.mean((residual_error_avg)**2)
MSE_test_avg = np.mean((forecast_error_avg)**2)
print('Mean Square Error of prediction errors for Average method: ', MSE_train_avg)
print('Mean Square Error of forecast errors for Average method: ', MSE_test_avg)
mean_pred_avg = np.mean(residual_error_avg)
var_pred_avg = np.var(residual_error_avg)
var_forecast_avg = np.var(forecast_error_avg)
print('Mean of prediction errors for Average method: ', mean_pred_avg)
print('Variance of prediction errors for Average method: ', var_pred_avg)
print('Variance of forecast errors for Average method: ', var_forecast_avg)

# Naive Method
def naive_method(t):
    return t

naive_train_pred = []
for i in range(0, len(train)-1):
    res = naive_method(train[i])
    naive_train_pred.append(res)

res = np.ones(len(test)) * train[-1]
naive_test_forecast1 = np.ones(len(test)) * res
naive_test_forecast = pd.DataFrame(naive_test_forecast1).set_index(test.index)
residual_error_naive = np.array(train[1:]) - np.array(naive_train_pred)
forecast_error_naive = test - naive_test_forecast1
MSE_train_naive = np.mean((residual_error_naive)**2)
MSE_test_naive = np.mean((forecast_error_naive)**2)
print('Mean Square Error of prediction errors for Naive method: ', MSE_train_naive)
print('Mean Square Error of forecast errors for Naive method: ', MSE_test_naive)
mean_pred_naive = np.mean(residual_error_naive)
var_pred_naive = np.var(residual_error_naive)
var_forecast_naive = np.var(forecast_error_naive)
print('Mean of prediction errors for Naive method: ', mean_pred_naive)
print('Variance of prediction errors for Naive method: ', var_pred_naive)
print('Variance of forecast errors for Naive method: ', var_forecast_naive)

# Drift method
def drift_method(t, h):
    y_hat_drift = t[len(t)-1] + h*((t[len(t)-1]-t[0])/(len(t) - 1))
    return y_hat_drift

drift_train_forecast=[]
for i in range(1, len(train)):
    if i == 1:
        drift_train_forecast.append(train[0])
    else:
        h = 1
        res = drift_method(train[0:i], h)
        drift_train_forecast.append(res)

drift_test_forecast1=[]
for h in range(1, len(test)+1):
    res = drift_method(train, h)
    drift_test_forecast1.append(res)

drift_test_forecast = pd.DataFrame(drift_test_forecast1).set_index(test.index)
residual_error_drift = np.array(train[1:]) - np.array(drift_train_forecast)
forecast_error_drift = np.array(test) - np.array(drift_test_forecast1)
MSE_train_drift = np.mean((residual_error_drift)**2)
MSE_test_drift = np.mean((forecast_error_drift)**2)
print('Mean Square Error of prediction errors for Drift method: ', MSE_train_drift)
print('Mean Square Error of forecast errors for Drift method: ', MSE_test_drift)
mean_pred_drift = np.mean(residual_error_drift)
var_pred_drift = np.var(residual_error_drift)
var_forecast_drift = np.var(forecast_error_drift)
print('Mean of prediction errors for Drift method: ', mean_pred_drift)
print('Variance of prediction errors for Drift method: ', var_pred_drift)
print('Variance of forecast errors for Drift method: ', var_forecast_drift)

#SES Method
def ses(t, damping_factor, l0):
    yhat4 = []
    yhat4.append(l0)
    for i in range(1, len(t)-1):
        res = damping_factor*(t[i]) + (1-damping_factor)*(yhat4[i-1])
        yhat4.append(res)
    return yhat4

l0 = train[0]
ses_train_pred = ses(train, 0.50, l0)
ses_test_forecast1 = np.ones(len(test)) * (0.5*(train[-1]) + (1-0.5)*(ses_train_pred[-1]))
ses_test_forecast = pd.DataFrame(ses_test_forecast1).set_index(test.index)
residual_error_ses = np.array(train[1:]) - np.array(ses_train_pred)
forecast_error_ses = np.array(test) - np.array(ses_test_forecast1)
MSE_train_SES = np.mean((residual_error_ses)**2)
MSE_test_SES = np.mean((forecast_error_ses)**2)
print('Mean Square Error of prediction errors for SES method: ', MSE_train_SES)
print('Mean Square Error of forecast errors for SES method: ', MSE_test_SES)
mean_pred_SES = np.mean(residual_error_ses)
var_pred_SES = np.var(residual_error_ses)
var_forecast_SES = np.var(forecast_error_ses)
print('Mean of prediction errors for SES method: ', mean_pred_SES)
print('Variance of prediction errors for SES method: ', var_pred_SES)
print('Variance of forecast errors for SES method: ', var_forecast_SES)

# SES Method using statsmodels for alpha=0.5
# ses_train = train.ewm(alpha=0.5, adjust=False).mean()  # Another way of doing it
ses_model1 = SimpleExpSmoothing(train)
ses_fitted_model1 = ses_model1.fit(smoothing_level=0.5, optimized=False)
ses_train_pred1 = ses_fitted_model1.fittedvalues.shift(-1)
ses_test_forecast1 = ses_fitted_model1.forecast(steps=len(test))
ses_test_forecast1 = pd.DataFrame(ses_test_forecast1).set_index(test.index)
MSE_test_SES1 = np.square(np.subtract(test.values, np.ndarray.flatten(ses_test_forecast1.values))).mean()

# Holt's Linear Trend
holtl_fitted_model = ets.ExponentialSmoothing(train, trend='add', damped=True, seasonal=None).fit()
holtl_train_pred = holtl_fitted_model.fittedvalues
holtl_test_forecast = holtl_fitted_model.forecast(steps=len(test))
holtl_test_forecast = pd.DataFrame(holtl_test_forecast).set_index(test.index)
residual_error_holtl = np.subtract(train.values, np.ndarray.flatten(holtl_train_pred.values))
forecast_error_holtl = np.subtract(test.values, np.ndarray.flatten(holtl_test_forecast.values))
MSE_train_holtl = np.mean((residual_error_holtl)**2)
MSE_test_holtl = np.mean((forecast_error_holtl)**2)
print("Mean Square Error of prediction errors for Holt's Linear method: ", MSE_train_holtl)
print("Mean Square Error of forecast errors for Holt's Linear method: ", MSE_test_holtl)
mean_pred_holtl = np.mean(residual_error_holtl)
var_pred_holtl = np.var(residual_error_holtl)
var_forecast_holtl = np.var(forecast_error_holtl)
print("Mean of prediction errors for Holt's Linear method: ", mean_pred_holtl)
print("Variance of prediction errors for Holt's Linear method: ", var_pred_holtl)
print("Variance of forecast errors for Holt's Linear method: ", var_forecast_holtl)

# Holt's Winter Seasonal Trend
holtw_fitted_model = ets.ExponentialSmoothing(train, trend='add', damped=True, seasonal='add', seasonal_periods=4).fit()
holtw_train_pred = holtw_fitted_model.fittedvalues
holtw_test_forecast = holtw_fitted_model.forecast(steps=len(test))
holtw_test_forecast = pd.DataFrame(holtw_test_forecast).set_index(test.index)
residual_error_holtw = np.subtract(train.values, np.ndarray.flatten(holtw_train_pred.values))
forecast_error_holtw = np.subtract(test.values, np.ndarray.flatten(holtw_test_forecast.values))
MSE_train_holtw = np.mean((residual_error_holtw)**2)
MSE_test_holtw = np.mean((forecast_error_holtw)**2)
print("Mean Square Error of prediction errors for Holt's Winter Seasonal method: ", MSE_train_holtw)
print("Mean Square Error of forecast errors for Holt's Winter Seasonal method: ", MSE_test_holtw)
mean_pred_holtw = np.mean(residual_error_holtw)
var_pred_holtw = np.var(residual_error_holtw)
var_forecast_holtw = np.var(forecast_error_holtw)
print("Mean of prediction errors for Holt's Winter Seasonal method: ", mean_pred_holtw)
print("Variance of prediction errors for Holt's Winter Seasonal method: ", var_pred_holtw)
print("Variance of forecast errors for Holt's Winter Seasonal method: ", var_forecast_holtw)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(test_forecast_avg, label='Average h-step prediction')
plt.xlabel('Time (Quarterly)')
plt.ylabel('Number of Sales')
plt.title('Average Method - Tute1')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(naive_test_forecast, label='Naive h-step prediction')
plt.xlabel('Time (Quarterly)')
plt.ylabel('Number of Sales')
plt.title('Naive Method - Tute1')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(drift_test_forecast, label='Drift h-step prediction')
plt.xlabel('Time (Quarterly)')
plt.ylabel('Number of Sales')
plt.title('Drift Method - Tute1')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(ses_test_forecast, label='Simple Exponential Smoothing h-step prediction')
plt.xlabel('Time (Quarterly)')
plt.ylabel('Number of Sales')
plt.title('SES Method - Tute1')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtl_test_forecast, label="Holt's Linear h-step prediction")
plt.xlabel('Time (Quarterly)')
plt.ylabel('Number of Sales')
plt.title("Holt's Linear Method - Tute1")
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtw_test_forecast, label="Holt's Winter Seasonal h-step prediction")
plt.xlabel('Time (Quarterly)')
plt.ylabel('Number of Sales')
plt.title("Holt's Winter Seasonal Method - Tute1")
plt.legend(loc='upper left')
plt.show()

# Auto_correlation for forecast errors and Q value for prediction errors and forecast errors
#Average Method
k = len(test)
lags = 15
avg_forecast_acf = cal_auto_corr(forecast_error_avg, lags)
Q_forecast_avg = k * np.sum(np.array(avg_forecast_acf[lags:])**2)
print('Q value of forecast errors for Average method: ', Q_forecast_avg)
plt.figure()
plt.stem(range(-(lags-1),lags), avg_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Average Method)')
plt.show()

# Naive method
k = len(test)
lags = 15
naive_forecast_acf = cal_auto_corr(forecast_error_naive, lags)
Q_forecast_naive = k * np.sum(np.array(naive_forecast_acf[lags:])**2)
print('Q value of forecast errors for Naive method: ', Q_forecast_naive)
plt.figure()
plt.stem(range(-(lags-1),lags), naive_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Naive Method)')
plt.show()

# Drift Method
k = len(test)
lags = 15
drift_forecast_acf = cal_auto_corr(forecast_error_drift, lags)
Q_forecast_drift = k * np.sum(np.array(drift_forecast_acf[lags:])**2)
print('Q value of forecast errors for Drift method: ', Q_forecast_drift)
plt.figure()
plt.stem(range(-(lags-1),lags), drift_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Drift Method)')
plt.show()

# SES method
k = len(test)
lags = 15
ses_forecast_acf = cal_auto_corr(forecast_error_ses, lags)
Q_forecast_SES = k * np.sum(np.array(ses_forecast_acf[lags:])**2)
print('Q value of forecast errors for SES method: ', Q_forecast_SES)
plt.figure()
plt.stem(range(-(lags-1), lags), ses_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (SES Method)')
plt.show()

# holt's linear method
k = len(test)
lags = 15
holtl_forecast_acf = cal_auto_corr(forecast_error_holtl, lags)
Q_forecast_holtl = k * np.sum(np.array(holtl_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Linear method: ", Q_forecast_holtl)
plt.figure()
plt.stem(range(-(lags-1), lags), holtl_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's Linear Method)")
plt.show()

# holt's Winter Seasonal method
k = len(test)
lags = 15
holtw_forecast_acf = cal_auto_corr(forecast_error_holtw, lags)
Q_forecast_holtw = k * np.sum(np.array(holtw_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Winter Seasonal method: ", Q_forecast_holtw)
plt.figure()
plt.stem(range(-(lags-1), lags), holtw_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's winter Seasonal Method)")
plt.show()

corr_avg = correlation_coefficent_cal(forecast_error_avg, test)
corr_naive = correlation_coefficent_cal(forecast_error_naive, test)
corr_drift = correlation_coefficent_cal(forecast_error_drift, test)
corr_ses = correlation_coefficent_cal(forecast_error_ses, test)
corr_holtl = correlation_coefficent_cal(forecast_error_holtl, test)
corr_holtw = correlation_coefficent_cal(forecast_error_holtw, test)
print("Correlation Coefficient between Forecast Error and Test set for Average Method: {}".format(corr_avg))
print("Correlation Coefficient between Forecast Error and Test set for Naive Method: {}".format(corr_naive))
print("Correlation Coefficient between Forecast Error and Test set for Drift Method: {}".format(corr_drift))
print("Correlation Coefficient between Forecast Error and Test set for SES Method: {}".format(corr_ses))
print("Correlation Coefficient between Forecast Error and Test set for Holt's Linear Method: {}".format(corr_holtl))
print("Correlation Coefficient between Forecast Error and Test set for Holt's winter seasonal Method: {}".format(corr_holtw))
d = {'Methods':['Average', 'Naive', 'Drift', 'SES', "HoltL", "HoltW"],
     'Q_val': [round(Q_forecast_avg, 2), round(Q_forecast_naive, 2), round(Q_forecast_drift,2), round(Q_forecast_SES,2), round(Q_forecast_holtl,2), round(Q_forecast_holtw,2)],
     'MSE(P)': [round(MSE_train_avg,2), round(MSE_train_naive,2), round(MSE_train_drift,2), round(MSE_train_SES,2), round(MSE_train_holtl,2), round(MSE_train_holtw,2)],
     'MSE(F)': [round(MSE_test_avg,2), round(MSE_test_naive,2), round(MSE_test_drift,2), round(MSE_test_SES,2), round(MSE_test_holtl,2), round(MSE_test_holtw,2)],
     'var(P)': [round(var_pred_avg,2), round(var_pred_naive,2), round(var_pred_drift,2), round(var_pred_SES,2), round(var_pred_holtl,2), round(var_pred_holtw,2)],
     'var(F)':[round(var_forecast_avg,2), round(var_forecast_naive,2), round(var_forecast_drift,2), round(var_forecast_SES,2), round(var_forecast_holtl,2), round(var_forecast_holtw,2)],
     'corrcoeff':[round(corr_avg,2), round(corr_naive,2), round(corr_drift,2), round(corr_ses,2), round(corr_holtl,2), round(corr_holtw,2)]}
df = pd.DataFrame(data=d)
df = df.set_index('Methods')
pd.set_option('display.max_columns', None)
print(df)


'----------------------------------------- daily-min-temperatures Dataset ---------------------------------------------'
df = pd.read_csv('daily-min-temperatures.csv', index_col='Date', parse_dates=True)
y = df['Temp']
train, test = train_test_split(y, shuffle=False, test_size=0.2)
h = len(test)
print('************************** daily-min-temperatures Dataset results ***********************')
#Average Method
def avg_method(train):
    y_hat_avg = np.mean(train)
    return y_hat_avg

train_pred_avg = []
for i in range(1,len(train)):
    res = avg_method(train.iloc[0:i])
    train_pred_avg.append(res)

test_forecast_avg1 = np.ones(len(test)) * avg_method(train)
test_forecast_avg = pd.DataFrame(test_forecast_avg1).set_index(test.index)
residual_error_avg = np.array(train[1:]) - np.array(train_pred_avg)
forecast_error_avg = test - test_forecast_avg1
MSE_train_avg = np.mean((residual_error_avg)**2)
MSE_test_avg = np.mean((forecast_error_avg)**2)
print('Mean Square Error of prediction errors for Average method: ', MSE_train_avg)
print('Mean Square Error of forecast errors for Average method: ', MSE_test_avg)
mean_pred_avg = np.mean(residual_error_avg)
var_pred_avg = np.var(residual_error_avg)
var_forecast_avg = np.var(forecast_error_avg)
print('Mean of prediction errors for Average method: ', mean_pred_avg)
print('Variance of prediction errors for Average method: ', var_pred_avg)
print('Variance of forecast errors for Average method: ', var_forecast_avg)

# Naive Method
def naive_method(t):
    return t

naive_train_pred = []
for i in range(0, len(train)-1):
    res = naive_method(train[i])
    naive_train_pred.append(res)

res = np.ones(len(test)) * train[-1]
naive_test_forecast1 = np.ones(len(test)) * res
naive_test_forecast = pd.DataFrame(naive_test_forecast1).set_index(test.index)
#print('h-step ahead Forecast for Naive Method', naive_test_forecast)
residual_error_naive = np.array(train[1:]) - np.array(naive_train_pred)
forecast_error_naive = test - naive_test_forecast1
MSE_train_naive = np.mean((residual_error_naive)**2)
MSE_test_naive = np.mean((forecast_error_naive)**2)
print('Mean Square Error of prediction errors for Naive method: ', MSE_train_naive)
print('Mean Square Error of forecast errors for Naive method: ', MSE_test_naive)
mean_pred_naive = np.mean(residual_error_naive)
var_pred_naive = np.var(residual_error_naive)
var_forecast_naive = np.var(forecast_error_naive)
print('Mean of prediction errors for Naive method: ', mean_pred_naive)
print('Variance of prediction errors for Naive method: ', var_pred_naive)
print('Variance of forecast errors for Naive method: ', var_forecast_naive)

# Drift method
def drift_method(t, h):
    y_hat_drift = t[len(t)-1] + h*((t[len(t)-1]-t[0])/(len(t) - 1))
    return y_hat_drift

drift_train_forecast=[]
for i in range(1, len(train)):
    if i == 1:
        drift_train_forecast.append(train[0])
    else:
        h = 1
        res = drift_method(train[0:i], h)
        drift_train_forecast.append(res)

drift_test_forecast1=[]
for h in range(1, len(test)+1):
    res = drift_method(train, h)
    drift_test_forecast1.append(res)

drift_test_forecast = pd.DataFrame(drift_test_forecast1).set_index(test.index)
residual_error_drift = np.array(train[1:]) - np.array(drift_train_forecast)
forecast_error_drift = np.array(test) - np.array(drift_test_forecast1)
MSE_train_drift = np.mean((residual_error_drift)**2)
MSE_test_drift = np.mean((forecast_error_drift)**2)
print('Mean Square Error of prediction errors for Drift method: ', MSE_train_drift)
print('Mean Square Error of forecast errors for Drift method: ', MSE_test_drift)
mean_pred_drift = np.mean(residual_error_drift)
var_pred_drift = np.var(residual_error_drift)
var_forecast_drift = np.var(forecast_error_drift)
print('Mean of prediction errors for Drift method: ', mean_pred_drift)
print('Variance of prediction errors for Drift method: ', var_pred_drift)
print('Variance of forecast errors for Drift method: ', var_forecast_drift)

#SES Method
def ses(t, damping_factor, l0):
    yhat4 = []
    yhat4.append(l0)
    for i in range(1, len(t)-1):
        res = damping_factor*(t[i]) + (1-damping_factor)*(yhat4[i-1])
        yhat4.append(res)
    return yhat4

l0 = train[0]
ses_train_pred = ses(train, 0.50, l0)
ses_test_forecast1 = np.ones(len(test)) * (0.5*(train[-1]) + (1-0.5)*(ses_train_pred[-1]))
ses_test_forecast = pd.DataFrame(ses_test_forecast1).set_index(test.index)
residual_error_ses = np.array(train[1:]) - np.array(ses_train_pred)
forecast_error_ses = np.array(test) - np.array(ses_test_forecast1)
MSE_train_SES = np.mean((residual_error_ses)**2)
MSE_test_SES = np.mean((forecast_error_ses)**2)
print('Mean Square Error of prediction errors for SES method: ', MSE_train_SES)
print('Mean Square Error of forecast errors for SES method: ', MSE_test_SES)
mean_pred_SES = np.mean(residual_error_ses)
var_pred_SES = np.var(residual_error_ses)
var_forecast_SES = np.var(forecast_error_ses)
print('Mean of prediction errors for SES method: ', mean_pred_SES)
print('Variance of prediction errors for SES method: ', var_pred_SES)
print('Variance of forecast errors for SES method: ', var_forecast_SES)

# SES Method using statsmodels for alpha=0.5
# ses_train = train.ewm(alpha=0.5, adjust=False).mean()  # Another way of doing it
ses_model1 = SimpleExpSmoothing(train)
ses_fitted_model1 = ses_model1.fit(smoothing_level=0.5, optimized=False)
ses_train_pred1 = ses_fitted_model1.fittedvalues.shift(-1)
ses_test_forecast1 = ses_fitted_model1.forecast(steps=len(test))
ses_test_forecast1 = pd.DataFrame(ses_test_forecast1).set_index(test.index)
MSE_test_SES1 = np.square(np.subtract(test.values, np.ndarray.flatten(ses_test_forecast1.values))).mean()

# Holt's Linear Trend
holtl_fitted_model = ets.ExponentialSmoothing(train, trend='add', damped=True, seasonal=None).fit()
holtl_train_pred = holtl_fitted_model.fittedvalues
holtl_test_forecast = holtl_fitted_model.forecast(steps=len(test))
holtl_test_forecast = pd.DataFrame(holtl_test_forecast).set_index(test.index)
residual_error_holtl = np.subtract(train.values, np.ndarray.flatten(holtl_train_pred.values))
forecast_error_holtl = np.subtract(test.values, np.ndarray.flatten(holtl_test_forecast.values))
MSE_train_holtl = np.mean((residual_error_holtl)**2)
MSE_test_holtl = np.mean((forecast_error_holtl)**2)
print("Mean Square Error of prediction errors for Holt's Linear method: ", MSE_train_holtl)
print("Mean Square Error of forecast errors for Holt's Linear method: ", MSE_test_holtl)
mean_pred_holtl = np.mean(residual_error_holtl)
var_pred_holtl = np.var(residual_error_holtl)
var_forecast_holtl = np.var(forecast_error_holtl)
print("Mean of prediction errors for Holt's Linear method: ", mean_pred_holtl)
print("Variance of prediction errors for Holt's Linear method: ", var_pred_holtl)
print("Variance of forecast errors for Holt's Linear method: ", var_forecast_holtl)

# Holt's Winter Seasonal Trend
holtw_fitted_model = ets.ExponentialSmoothing(train, trend='add', damped=True, seasonal='add', seasonal_periods=365).fit()
holtw_train_pred = holtw_fitted_model.fittedvalues
holtw_test_forecast = holtw_fitted_model.forecast(steps=len(test))
holtw_test_forecast = pd.DataFrame(holtw_test_forecast).set_index(test.index)
residual_error_holtw = np.subtract(train.values, np.ndarray.flatten(holtw_train_pred.values))
forecast_error_holtw = np.subtract(test.values, np.ndarray.flatten(holtw_test_forecast.values))
MSE_train_holtw = np.mean((residual_error_holtw)**2)
MSE_test_holtw = np.mean((forecast_error_holtw)**2)
print("Mean Square Error of prediction errors for Holt's Winter Seasonal method: ", MSE_train_holtw)
print("Mean Square Error of forecast errors for Holt's Winter Seasonal method: ", MSE_test_holtw)
mean_pred_holtw = np.mean(residual_error_holtw)
var_pred_holtw = np.var(residual_error_holtw)
var_forecast_holtw = np.var(forecast_error_holtw)
print("Mean of prediction errors for Holt's Winter Seasonal method: ", mean_pred_holtw)
print("Variance of prediction errors for Holt's Winter Seasonal method: ", var_pred_holtw)
print("Variance of forecast errors for Holt's Winter Seasonal method: ", var_forecast_holtw)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(test_forecast_avg, label='Average h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Daily Minimum Temperature')
plt.title('Average Method - daily-min-temperatures')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(naive_test_forecast, label='Naive h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Daily Minimum Temperature')
plt.title('Naive Method - daily-min-temperatures')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(drift_test_forecast, label='Drift h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Daily Minimum Temperature')
plt.title('Drift Method - daily-min-temperatures')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(ses_test_forecast, label='Simple Exponential Smoothing h-step prediction')
plt.xlabel('Time (Daily)')
plt.ylabel('Daily Minimum Temperature')
plt.title('SES Method - daily-min-temperatures')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtl_test_forecast, label="Holt's Linear h-step prediction")
plt.xlabel('Time (Daily)')
plt.ylabel('Daily Minimum Temperature')
plt.title("Holt's Linear Method - daily-min-temperatures")
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train, label='Training set')
ax.plot(test, label='Testing set')
ax.plot(holtw_test_forecast, label="Holt's Winter Seasonal h-step prediction")
plt.xlabel('Time (Daily)')
plt.ylabel('Daily Minimum Temperature')
plt.title("Holt's Winter Seasonal Method - daily-min-temperatures")
plt.legend(loc='upper left')
plt.show()

# Auto_correlation for forecast errors and Q value for prediction errors and forecast errors
#Average Method
k = len(test)
lags = 15
avg_forecast_acf = cal_auto_corr(forecast_error_avg, lags)
Q_forecast_avg = k * np.sum(np.array(avg_forecast_acf[lags:])**2)
print('Q value of forecast errors for Average method: ', Q_forecast_avg)
plt.figure()
plt.stem(range(-(lags-1),lags), avg_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Average Method)')
plt.show()

# Naive method
k = len(test)
lags = 15
naive_forecast_acf = cal_auto_corr(forecast_error_naive, lags)
Q_forecast_naive = k * np.sum(np.array(naive_forecast_acf[lags:])**2)
print('Q value of forecast errors for Naive method: ', Q_forecast_naive)
plt.figure()
plt.stem(range(-(lags-1),lags), naive_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Naive Method)')
plt.show()

# Drift Method
k = len(test)
lags = 15
drift_forecast_acf = cal_auto_corr(forecast_error_drift, lags)
Q_forecast_drift = k * np.sum(np.array(drift_forecast_acf[lags:])**2)
print('Q value of forecast errors for Drift method: ', Q_forecast_drift)
plt.figure()
plt.stem(range(-(lags-1),lags), drift_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (Drift Method)')
plt.show()

# SES method
k = len(test)
lags = 15
ses_forecast_acf = cal_auto_corr(forecast_error_ses, lags)
Q_forecast_SES = k * np.sum(np.array(ses_forecast_acf[lags:])**2)
print('Q value of forecast errors for SES method: ', Q_forecast_SES)
plt.figure()
plt.stem(range(-(lags-1), lags), ses_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error (SES Method)')
plt.show()

# holt's linear method
k = len(test)
lags = 15
holtl_forecast_acf = cal_auto_corr(forecast_error_holtl, lags)
Q_forecast_holtl = k * np.sum(np.array(holtl_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Linear method: ", Q_forecast_holtl)
plt.figure()
plt.stem(range(-(lags-1), lags), holtl_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's Linear Method)")
plt.show()

# holt's Winter Seasonal method
k = len(test)
lags = 15
holtw_forecast_acf = cal_auto_corr(forecast_error_holtw, lags)
Q_forecast_holtw = k * np.sum(np.array(holtw_forecast_acf[lags:])**2)
print("Q value of forecast errors for Holt's Winter Seasonal method: ", Q_forecast_holtw)
plt.figure()
plt.stem(range(-(lags-1), lags), holtw_forecast_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title("Autocorrelation plot for Forecast Error (Holt's winter Seasonal Method)")
plt.show()

corr_avg = correlation_coefficent_cal(forecast_error_avg, test)
corr_naive = correlation_coefficent_cal(forecast_error_naive, test)
corr_drift = correlation_coefficent_cal(forecast_error_drift, test)
corr_ses = correlation_coefficent_cal(forecast_error_ses, test)
corr_holtl = correlation_coefficent_cal(forecast_error_holtl, test)
corr_holtw = correlation_coefficent_cal(forecast_error_holtw, test)
print("Correlation Coefficient between Forecast Error and Test set for Average Method: {}".format(corr_avg))
print("Correlation Coefficient between Forecast Error and Test set for Naive Method: {}".format(corr_naive))
print("Correlation Coefficient between Forecast Error and Test set for Drift Method: {}".format(corr_drift))
print("Correlation Coefficient between Forecast Error and Test set for SES Method: {}".format(corr_ses))
print("Correlation Coefficient between Forecast Error and Test set for Holt's Linear Method: {}".format(corr_holtl))
print("Correlation Coefficient between Forecast Error and Test set for Holt's winter seasonal Method: {}".format(corr_holtw))
d = {'Methods':['Average', 'Naive', 'Drift', 'SES', "HoltL", "HoltW"],
     'Q_val': [round(Q_forecast_avg, 2), round(Q_forecast_naive, 2), round(Q_forecast_drift,2), round(Q_forecast_SES,2), round(Q_forecast_holtl,2), round(Q_forecast_holtw,2)],
     'MSE(P)': [round(MSE_train_avg,2), round(MSE_train_naive,2), round(MSE_train_drift,2), round(MSE_train_SES,2), round(MSE_train_holtl,2), round(MSE_train_holtw,2)],
     'MSE(F)': [round(MSE_test_avg,2), round(MSE_test_naive,2), round(MSE_test_drift,2), round(MSE_test_SES,2), round(MSE_test_holtl,2), round(MSE_test_holtw,2)],
     'var(P)': [round(var_pred_avg,2), round(var_pred_naive,2), round(var_pred_drift,2), round(var_pred_SES,2), round(var_pred_holtl,2), round(var_pred_holtw,2)],
     'var(F)':[round(var_forecast_avg,2), round(var_forecast_naive,2), round(var_forecast_drift,2), round(var_forecast_SES,2), round(var_forecast_holtl,2), round(var_forecast_holtw,2)],
     'corrcoeff':[round(corr_avg,2), round(corr_naive,2), round(corr_drift,2), round(corr_ses,2), round(corr_holtl,2), round(corr_holtw,2)]}
df = pd.DataFrame(data=d)
df = df.set_index('Methods')
pd.set_option('display.max_columns', None)
print(df)
