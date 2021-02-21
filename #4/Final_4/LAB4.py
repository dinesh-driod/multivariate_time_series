import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Autocorrelation import *

#Using the Python program and using only the “numpy” and “matplotlib” library perform the following tasks:
# %%============================================================================================================
# 1: Let suppose a time series dataset is given as below. Without a help of Python and using the
# average forecast method perform one-step ahead prediction and fill out the table. To perform
# the correct cross-validation, start with two observations {y1, y2} and predict y3 (you can now
# generate the first error). Add observations by one {y1, y2, y3} and predict y4 (you can now
# generate the second error). Continue this through the dataset. Then calculate the MSE of the 1-
# step prediction and MSE of h-step forecast.
# %%------------------------------------------------------------------------------------------------------------
yt = [112, 118, 132, 129, 121, 135, 148, 136, 119]
yf = [104, 118, 115, 126, 141]
# see lab report

# %%============================================================================================================
# 2: Write a python code that perform the task in step 1.
# Plot the test set, training set and the h-step forecast in one graph with different marker/color.
# Add an appropriate title, legend, x-label, y-label to each graph.
# No need to include the 1-step prediction in this graph.
# %%============================================================================================================
print('\n')
print(20 * "-" + "FORECAST METHOD| AVERAGE" + 20 * "-")
def avg_method(yt):
    return np.mean(yt)

yhat1 = []
for i in range(1,len(yt)):
    res = avg_method(yt[0:i])
    yhat1.append(res)
print("Average method 1-step prediction: ", yhat1)

y_fr = np.ones(len(yf)) * (np.mean(yt))
print("Average method h-step prediction: ", y_fr)

plt.figure(figsize=(16,10))
plt.plot(range(1, len(yt)+1), yt, label='Training set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), y_fr, label='Average h-step prediction')
plt.xlabel('TIME')
plt.ylabel('Y')
plt.title('AVERAGE METHOD: Training, Testing and Forecast values')
plt.legend()
plt.show()

# %%============================================================================================================
# 3: Using python, calculate the MSE of prediction errors and the forecast errors.
# Display the result on the console.
# %%------------------------------------------------------------------------------------------------------------
residual_error_avg = np.array(yt[1:]) - np.array(yhat1)
forecast_error_avg = yf - y_fr
MSE_train_avg = np.mean((residual_error_avg)**2)
MSE_test_avg = np.mean((forecast_error_avg)**2)

print('Mean Square Error of prediction errors for Average method: ', MSE_train_avg)
print('Mean Square Error of forecast errors for Average method: ', MSE_test_avg)


# %%============================================================================================================
# 4: Using python, calculate the variance of prediction error and the variance of forecast error.
# Display the result.
# %%------------------------------------------------------------------------------------------------------------
mean_pred_avg = np.mean(residual_error_avg)
var_pred_avg = np.var(residual_error_avg)
var_forecast_avg = np.var(forecast_error_avg)
print('Mean of prediction errors for Average method: ', mean_pred_avg)
print('Variance of prediction errors for Average method: ', var_pred_avg)
print('Variance of forecast errors for Average method: ', var_forecast_avg)


# %%============================================================================================================
# 5: Calculate the Q value for this estimate and display the Q-value on the console. (# of lags = 8)
# %%------------------------------------------------------------------------------------------------------------
k = len(yt)
lags = len(residual_error_avg)
avg_acf = cal_auto_corr(residual_error_avg, lags)
print(avg_acf)
Q_avg = k * np.sum(np.array(avg_acf[8:])**2)
print('Q value for Average method: ', Q_avg)
print('\n')
# %%============================================================================================================
# 6: Repeat step 1 through 5 with the Naïve method.
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "FORECAST METHOD| NAIVE" + 20 * "-")
def naive_method(yt):
    return yt

yhat2 = []
for i in range(0, len(yt)-1):
    res = naive_method(yt[i])
    yhat2.append(res)

print("Naive method 1-step prediction: ", yhat2)
y_fr = np.ones(len(yf)) * yt[-1]
print("Naive method h-step prediction: ", y_fr)


residual_error_naive = np.array(yt[1:]) - np.array(yhat2)
forecast_error_naive = yf - y_fr
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

plt.figure(figsize=(16,10))
plt.plot(range(1, len(yt)+1), yt, label='Training set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), y_fr, label='Naive h-step prediction')
plt.xlabel('time')
plt.ylabel('y')
plt.title('Training, Testing and Forecast values for Naive Method')
plt.legend()
plt.show()

k = len(yt)
lags = len(residual_error_naive)
naive_acf = cal_auto_corr(residual_error_naive, lags)
Q_naive = k * np.sum(np.array(naive_acf[8:])**2)
print('Q value for Naive method: ', Q_naive)
print('\n')
# %%============================================================================================================
# 7: Repeat step 1 through 5 with the drift method.
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "FORECAST METHOD| DRIFT" + 20 * "-")
yhat3 = []
def drift_method(t, h):
    res = t[len(t)-1] + h*((t[len(t)-1]-t[0])/(len(t) - 1))
    return res

for i in range(1, len(yt)):
    if i==1:
        yhat3.append(yt[0])
    else:
        h = 1
        res = drift_method(yt[0:i], h)
        yhat3.append(res)

print("Drift method 1-step prediction: ", yhat3)

y_fr = []
for h in range(1, len(yf)+1):
    res = drift_method(yt,h)
    y_fr.append(res)
print("Drift method h-step prediction: ", y_fr)
residual_error_drift = np.array(yt[1:]) - np.array(yhat3)
forecast_error_drift = np.array(yf) - np.array(y_fr)
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

plt.figure(figsize=(16,10))
plt.plot(range(1, len(yt)+1), yt, label='Training set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), y_fr, label='Drift h-step prediction')
plt.xlabel('time')
plt.ylabel('y')
plt.title('Training, Testing and Forecast values for Drift Method')
plt.legend()
plt.show()

k = len(yt)
lags = len(residual_error_drift)
drift_acf = cal_auto_corr(residual_error_drift, lags)
Q_drift = k * np.sum(np.array(drift_acf[8:])**2)
print('Q value for Drift method: ', Q_drift)
print('\n')
# %%============================================================================================================
#8: Repeat step 1 through 5 with the simple exponential method.
# Consider alfa = 0.5 and the initial condition to be the first sample in the training set.
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "FORECAST METHOD| SIMPLE EXPONENTIAL METHOD" + 20 * "-")

def ses(t, damping_factor, l0):
    yhat4 = []
    yhat4.append(l0)
    for i in range(1, len(t)-1):
        res = damping_factor*(t[i]) + (1-damping_factor)*(yhat4[i-1])
        yhat4.append(res)
    return yhat4

l0 = yt[0]
ses_0 = ses(yt, 0, l0)
ses_25 = ses(yt, 0.25, l0)
ses_50 = ses(yt, 0.50, l0)
ses_75 = ses(yt, 0.75, l0)
ses_99 = ses(yt, 0.99, l0)

print("SES method 1-step prediction with damping factor 0.5: ", ses_50)
ses_fr_50 = np.ones(len(yf)) * (0.5*(yt[-1]) + (1-0.5)*(ses_50[-1]))
print("SES method h-step prediction with damping factor 0.5: ", ses_fr_50)

residual_error_ses = np.array(yt[1:]) - np.array(ses_50)
forecast_error_ses = np.array(yf) - np.array(ses_fr_50)
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

# Plot of test set, training set and the h-step forecast
plt.figure(figsize=(16,10))
plt.plot(range(1, len(yt)+1), yt, label='Training set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
plt.plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), ses_fr_50, label='SES h-step prediction')
plt.xlabel('time')
plt.ylabel('y')
plt.title('Training, Testing and Forecast with alfa=0.5 for Simple Exponential Smoothing Method')
plt.legend()
plt.show()

#Q-VALUE
k = len(yt)
lags = len(residual_error_ses)
ses_acf = cal_auto_corr(residual_error_ses, lags)
Q_SES = k * np.sum(np.array(ses_acf[8:])**2)
print('Q value for SES method: ', Q_SES)
print('\n')
# %%============================================================================================================
#9: Using SES method, plot the test set, training set and the h-step forecast in one graph for alfa = 0,
# 0.25, 0.75 and 0.99. You can use a subplot 2x2.
# Add an appropriate title, legend, x-label, y-label to each graph.
# No need to include the 1-step prediction in this graph
# %%------------------------------------------------------------------------------------------------------------
ses_fr_0 = np.ones(len(yf)) * (0.5*(yt[-1]) + (1-0.5)*(ses_0[-1]))
ses_fr_25 = np.ones(len(yf)) * (0.5*(yt[-1]) + (1-0.5)*(ses_25[-1]))
ses_fr_75 = np.ones(len(yf)) * (0.5*(yt[-1]) + (1-0.5)*(ses_75[-1]))
ses_fr_99 = np.ones(len(yf)) * (0.5*(yt[-1]) + (1-0.5)*(ses_99[-1]))

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
ax[0,0].plot(range(1, len(yt)+1), yt, label='Training set')
ax[0,0].plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
ax[0,0].plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), ses_fr_0, label='SES h-step prediction')
ax[0,0].set_title('Training, Testing and Forecast with alfa=0 for Simple Exponential Smoothing Method')
ax[0,1].plot(range(1, len(yt)+1), yt, label='Training set')
ax[0,1].plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
ax[0,1].plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), ses_fr_25, label='SES h-step prediction')
ax[0,1].set_title('Training, Testing and Forecast with alfa=0.25 for Simple Exponential Smoothing Method')
ax[1,0].plot(range(1, len(yt)+1), yt, label='Training set')
ax[1,0].plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
ax[1,0].plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), ses_fr_75, label='SES h-step prediction')
ax[1,0].set_title('Training, Testing and Forecast with alfa=0.75 for Simple Exponential Smoothing Method')
ax[1,1].plot(range(1, len(yt)+1), yt, label='Training set')
ax[1,1].plot(range(len(yt)+1, (len(yt)+1) + len(yf)), yf, label='Testing set')
ax[1,1].plot(range(len(yt)+1, (len(yt)+1) + len(y_fr)), ses_fr_99, label='SES h-step prediction')
ax[1,1].set_title('Training, Testing and Forecast with alfa=0.99 for Simple Exponential Smoothing Method')
plt.xlabel('time')
plt.ylabel('y')
ax[0,0].legend(loc="upper right")
ax[0,1].legend(loc="upper right")
ax[1,0].legend(loc="upper right")
ax[1,1].legend(loc="upper right")
plt.show()

# %%============================================================================================================
#10: Create a table and compare the four forecast method above by displaying, Q values, MSE, mean
# of prediction errors, variance of prediction errors.
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "COMPARISION OF FORECAST METHODS" + 20 * "-")
d = {'Method':['Average', 'Naive', 'Drift', 'Simple Exponential Smoothing'],
     'Q_val': [Q_avg, Q_naive, Q_drift, Q_SES],
     'MSE_pred': [MSE_train_avg, MSE_train_naive, MSE_train_drift, MSE_train_SES],
     'MSE_forecast': [MSE_test_avg, MSE_test_naive, MSE_test_drift, MSE_test_SES],
     'Mean_pred': [mean_pred_avg, mean_pred_naive, mean_pred_drift, mean_pred_SES],
     'variance_pred': [var_pred_avg, var_pred_naive, var_pred_drift, var_pred_SES],
     'variance_forecast':[var_forecast_avg, var_forecast_naive, var_forecast_drift, var_forecast_SES]}
df = pd.DataFrame(data=d)
df = df.set_index('Method')
pd.set_option('display.max_columns', None)
print(df.head())

# %%============================================================================================================
#11: Using the python program developed in the previous LAB, plot the ACF of prediction errors.
# %%------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(16,10))
plt.stem(range(-(lags-1),lags), avg_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Prediction Error (Average Method)')
plt.show()

plt.figure(figsize=(16,10))
plt.stem(range(-(lags-1),lags), naive_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Prediction Error (Naive Method)')
plt.show()

plt.figure(figsize=(16,10))
plt.stem(range(-(lags-1),lags), drift_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Prediction Error (Drift Method)')
plt.show()

plt.figure(figsize=(16,10))
plt.stem(range(-(lags-1),lags), ses_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Prediction Error (SES Method)')
plt.show()

# %%============================================================================================================
#12: Compare the above 4 methods by looking at the variance of prediction error versus the variance of forecast error
# and pick the best estimator. Justify your answer.
# %%------------------------------------------------------------------------------------------------------------
