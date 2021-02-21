import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Using the Python program and using only the “numpy” and “matplotlib” library perform the following tasks:
# %%============================================================================================================
# 1: Let suppose y vectors is given as y(t) = [3, 9, 27, 81,243].
# Without use of python or any other computer program, manually calculate the 𝜏0, 𝜏1, 𝜏2, 𝜏3, 𝜏4.
# Display the ACF (two sided) on a graph (no python).
# %%------------------------------------------------------------------------------------------------------------
# see lab report


# %%============================================================================================================
# 2: Using Python program, create a white noise with zero mean and standard deviation of 1 and 1000 samples.
# Plot the generated WN versus number of samples. Plot the histogram of generated WN.
# Calculate the mean and std of generated WN.
# You can use the following command to generate
# WN(0,1): (import numpy as np, T # of samples)
# np.random.normal(mean, std, size=T)
# %%============================================================================================================
# Number of Samples
T = 1000
# Mean
m = 0
# Standard Deviation
s = 1

Y = np.random.normal(m,s, size=T)
plt.figure()
plt.plot(Y, label = 'White Noise')
plt.xlabel('Number of Samples')
plt.ylabel('Magnitude')
plt.title('White noise with {} Samples'.format(T))
plt.show()

plt.figure()
plt.hist(Y)
plt.title('Histogram plot White Noise with {} Samples'.format(T))
plt.show()

print("The Mean of white noise:",np.mean(Y))
print("The Standard Deviation of White Noise:",np.std(Y))

# %%============================================================================================================
# 3: Write a python code to estimate Autocorrelation Function.
# Note: You need to use the equation (1) given in lecture 4.
# %%------------------------------------------------------------------------------------------------------------
def auto_corr(y,k):
    T = len(y)
    y_mean = np.mean(y)
    res_num = 0
    res_den = 0
    for t in range(k,T):
        res_num += (y[t] - y_mean) * (y[t-k] - y_mean)

    for t in range(0,T):
        res_den += (y[t] - y_mean)**2

    res = res_num/res_den
    return res

def auto_corr_cal(y,k):
    res = []
    for t in range(0,k):
        result = auto_corr(y,t)
        res.append(result)
    return res

# %%============================================================================================================
# 3: a. Plot the ACF for the generated data in step 3.
# The ACF needs to be plotted using “stem” command.
# b. Write down your observations about the ACF plot, histogram, and the time plot of the
# generated WN
# %%------------------------------------------------------------------------------------------------------------
#a.

k = 20
acfcal = auto_corr_cal(Y,k)
acfplotvals = acfcal[::-1] + acfcal[1:]
plt.figure()
plt.stem(range(-(k - 1), k), acfplotvals)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF of Generated White Noise∼(0,1)')
plt.show()
# %%============================================================================================================
# 3: b. Write down your observations about the ACF plot, histogram, and the time plot of the
# generated WN
# %%------------------------------------------------------------------------------------------------------------
'''
Auto correlation of white noise have a strong peak at 0 and absolutely zero for all other lags which is basically an 
impulse. 
Histogram as an gaussian distribution. 
From the time plot we can see the variables are independent and identically distributed with a mean of zero. 
This means that all variables have the same variance (sigma^2) and each value has a zero correlation with all 
other values in the series.

'''

# %%============================================================================================================
# 4: Load the time series dataset tute1.csv (from LAB#1)
# %%------------------------------------------------------------------------------------------------------------
k = 20
df = pd.read_csv('tute1.csv')
date_rng = pd.date_range(start='3/1/1981', end='3/1/2006', freq='Q')

# %%============================================================================================================
# 4 a. Using python code written in the previous step,
# plot the ACF for the “Sales” and “Sales” versus time next to each other. You can use subplot command.
# %%------------------------------------------------------------------------------------------------------------
Salesacf = auto_corr_cal(df['Sales'],k)
Salesacfplotvals = Salesacf[::-1] + Salesacf[1:]
plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(range(-(k-1),k), Salesacfplotvals)
ax1.set(xlabel = 'Lags', ylabel = 'Magnitude', title = 'ACF for the "Sales"')
ax2.plot(date_rng, df['Sales'], label = 'Sales')
ax2.set(xlabel = 'Quarterly Data', ylabel = 'Sales in $', title = 'Time series plot of Sales data')
plt.show()

# %%============================================================================================================
# 4 b. Using python code written in the previous step,
# plot the ACF for the “Sales” and “Sales” versus time next to each other. You can use subplot command.
# %%------------------------------------------------------------------------------------------------------------
AdBudgetacf = auto_corr_cal(df['AdBudget'],k)

AdBudgetacfplotvals = AdBudgetacf[::-1] + AdBudgetacf[1:]
plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(range(-(k-1),k), AdBudgetacfplotvals)
ax1.set(xlabel = 'Lags', ylabel = 'Magnitude', title = 'ACF for the "AdBudget"')
ax2.plot(date_rng, df['AdBudget'], label = 'AdBudget')
ax2.set(xlabel = 'Quarterly Data', ylabel = 'AdBudget in $', title = 'Time series plot of AdBudget data')
plt.show()

# %%============================================================================================================
# 4 c. Using python code written in the previous step, plot the ACF for the “GDP” and “GDP”
# versus time next to each other. You can use subplot command.
# %%------------------------------------------------------------------------------------------------------------
GDPacf = auto_corr_cal(df['GDP'],k)

GDPacfplotvals = GDPacf[::-1] + GDPacf[1:]
plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(range(-(k-1),k), GDPacfplotvals)
ax1.set(xlabel = 'Lags', ylabel = 'Magnitude', title = 'ACF for the "GDP"')
ax2.plot(date_rng, df['GDP'], label = 'GDP')
ax2.set(xlabel = 'Quarterly Data', ylabel = 'GDP in $', title = 'Time series plot of GDP data')
plt.show()
# %%============================================================================================================
# 4 d. Write down your observations about the correlation between stationary and nonstationary time series
# (if there is any) and autocorrelation function?
# %%------------------------------------------------------------------------------------------------------------
'''
In stationary (time) series, statistical properties such as the mean, variance and autocorrelation are all 
constant over time where as in a non-stationary series statistical properties change over time.
For a stationary time series, the ACF will drop to zero relatively quickly, 
while the ACF of non-stationary data decreases slowly. 
'''
# %%============================================================================================================
# 4 e. The number lags used for this question is 20.
# %%------------------------------------------------------------------------------------------------------------
# Initalized K as 20