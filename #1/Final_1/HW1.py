#Homework # 1- Nonstationary process and removing trend from time series data (1st , 2nd differencing and logarithmic transformation)
# Using the Python program and the required libraries perform the following tasks:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# %%============================================================================================================
# 1: Load the time series data called AirPassengers.csv.
# %%------------------------------------------------------------------------------------------------------------
data = pd.read_csv("AirPassengers.csv")
print("Shape of AirPassengers:",data.shape)
print('\n')

# %%============================================================================================================
# 2: This date relates to the air passengers between 1949-1960.
# %%------------------------------------------------------------------------------------------------------------
print("DataType of AirPassengers:", data.dtypes)
print('\n')
for i in data.columns:
        print(i)
print('\n')
# %%============================================================================================================
# 3: Write a Python code to display the first 5 rows of the data on the console.
# %%------------------------------------------------------------------------------------------------------------
#Creating the 'Date' as Index for data and viewing the dataset
data.Month = pd.to_datetime(data.Month)
print(data.head(5))
print('\n')
print(data.tail(5))
print('\n')
# %%============================================================================================================
# 4: Explore the dataset by plotting the entire dataset, where you can learn more about the data set pattern
# (trend, seasonality, cyclic, …). Add the label to the horizontal and vertical axis as Month and Sales Number.
# Add the title as “Air passengers Dataset without differencing”.
# Add an appropriate legend to your plot.
# Do you see any trend, seasonality, or cyclical behavior in the plotted dataset?
# If yes, what is it?
# %%------------------------------------------------------------------------------------------------------------
#Visualizing the Time Series plot for the number of Air Passengers
plt.figure(figsize=(16,10))
plt.plot(data['Month'], data['#Passengers'], label = 'Passengers')
plt.xlabel("Month")
plt.ylabel("Number of Passengers")
plt.title('Air passengers Dataset without differencing')
plt.legend()
plt.show()
'''
It’s clear from the plot that there is an overall increase in the trend,with some seasonality in it.
'''
# %%============================================================================================================
# 5: Run an ADF-test and check if the dataset is stationary or not. Is the dataset non-stationary? Justify your answer.
# Calculate the average over the entire dataset and show the average plot.
# %%------------------------------------------------------------------------------------------------------------
def ADFcal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" %result[1])
    print("Critical values :")
    for key,value in result[4].items():
        print("\t%s: %.3f" % (key, value))
    print()

ADFcal(data['#Passengers'])
passavg = []
for i in range(1,145):
    k = pd.read_csv('AirPassengers.csv').head(i)
    passavg.append(np.mean(k['#Passengers']))

plt.figure(figsize=(16,10))
plt.plot(data['Month'], passavg, label = 'Average Passengers')
plt.xlabel("Month")
plt.ylabel("Number of Average Passengers")
plt.title('Average Air passengers Dataset without differencing')
plt.legend()
plt.show()
'''
From above ADF test, we fail to reject the null hypothesis, since p-value is greater than 0.05
Below we took log transformation to make our Time series stationary and plotted visual for it
We found graph upward trending over time with seasonality
'''
# %%============================================================================================================
# 6: If the answer to the previous question is non-stationary,
# write a python code that detrend the dataset by 1st difference transformation. Plot the detrended dataset.
# %%------------------------------------------------------------------------------------------------------------
# 1st difference transformation
diff1 = data['#Passengers'].diff()
plt.figure(figsize=(16,10))
plt.plot(data['Month'], diff1, label = 'Passengers')
plt.xlabel("Month")
plt.ylabel("Number of Passengers")
plt.title('Air passengers Dataset after 1st difference transformation')
plt.legend()
plt.show()

ADFcal(diff1[1:])
# %%============================================================================================================
# 7: Is the detrended dataset stationary?
# Justify your answer by running an ADF-test. Plot the average and variance over the entire dataset.
# %%------------------------------------------------------------------------------------------------------------
#Yes it is.
ADFcal(diff1[1:])
# %%============================================================================================================
# 8:If the first differencing did not make the dataset to be stationary, then try 2nd differencing and repeat 7.
# %%------------------------------------------------------------------------------------------------------------
#Yes it did not make the dataset to stationary.
pass1avg, pass1var = [], []
for i in range(1,len(diff1)):
    pass1avg.append(np.mean(diff1[:i]))
    pass1var.append(np.var(diff1[:i]))
# %%============================================================================================================
# 9: If the 2nd differencing did not make the dataset to be stationary,
# then perform the 1st differencing followed by logarithmic transformation. The repeat step 7.
# %%------------------------------------------------------------------------------------------------------------
# 2nd difference transformation
diff2 = diff1.diff()
plt.figure(figsize=(16,10))
plt.plot(data['Month'], diff2, label = 'Passengers')
plt.xlabel("Month")
plt.ylabel("Number of Passengers")
plt.title('Air passengers Dataset after 2nd difference transformation')
plt.legend()
plt.show()

# %%============================================================================================================
# 10: If the procedures in step 6, 8 & 9 did not make the dataset to be stationary
# (pass the ADF-test with 95% or more confidence interval) then stop.
# %%------------------------------------------------------------------------------------------------------------
ADFcal(diff2[2:])
# %%============================================================================================================
# 11: Write a report and answer all the above questions. Include the required graphs into your report.
# %%------------------------------------------------------------------------------------------------------------
#Created Solution Report with above code and added all relavant graphs



