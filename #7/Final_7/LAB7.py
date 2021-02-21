import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from Autocorrelation import *

# %%============================================================================================================
# 1: Using the Python program and appropriate libraries perform the following tasks:
# Let consider an AR(2) process as 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) − 0.2𝑦(𝑡 − 2) = 𝑒(𝑡)
# Where 𝑒(𝑡) is a WN (1,2).
# %%------------------------------------------------------------------------------------------------------------
# a. Find the theoretical mean and variance of y(t). (no need to use python).
# b. Using python, create a for loop that simulates above process for 1000 samples. Assume all initial conditions to be zero.
# c. Using the generated samples in part b and numpy package, find the experimental mean
# and variance. Compare your answer with part a. Write down your observations.
# d. Plot the y(t) with respect to number of samples.
# e. Using the python code, developed in previous labs, calculate autocorrelations for 20 lags
# and plot them versus number of lags. Write down your observation about the ACF of
# above process.
# f. Display the first 5 values of y(t) at the console.
# g. Apply the ADF-test and check if this is a stationary process. Explain your answer
# %%------------------------------------------------------------------------------------------------------------
def ADFcal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" %result[1])
    print("Critical values :")
    for key,value in result[4].items():
        print("\t%s: %.3f" % (key, value))
    print()
# print("#### ADF test on Samples ####")
print(20 * "-" + "ARPROCESS" + 20 * "-")
T = 1000
mean = 1
std = np.sqrt(2)
np.random.seed(10)
e = np.random.normal(mean, std, size=T)
# print('The mean of WN is ', np.mean(e), 'The variance of the WN is ', np.var(e))
y = np.zeros(len(e))
for t in range(len(e)):
    if t == 0:
        y[t] = e[t]
    elif t == 1:
        y[t] = 0.5*y[t-1] + e[t]
    else:
        y[t] = 0.5*y[t-1] + 0.2*y[t-2] + e[t]

print('The mean of AR(2) is ', np.mean(y), 'The variance of the AR(2) is ', np.var(y))

plt.figure()
plt.plot(y)
plt.xlabel('Number of Samples')
plt.ylabel('y(t)')
plt.title('y(t) vs number of samples')
plt.show()

k = 20
result = cal_auto_corr(y, k)
plt.figure()
plt.stem(range(-(k-1),k), result, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot of AR')
plt.show()

print('First five values of y(t) are {}'.format(y[0:5]))

ADFcal(y)

# %%============================================================================================================
#2. Using the “scipy” python package and “dlsim” command, simulate the AR(2) process in question
# a. Display the first 5 values of y(t) at the console.
# b. Show that your answer to the previous part is identical to the answer in part d of previous question.
# %%------------------------------------------------------------------------------------------------------------

np.random.seed(10)
e = np.random.normal(mean, std, size=T)
num = [1,0,0]
den = [1, -0.5, -0.2]
sys = (num, den, 1)
_, y =signal.dlsim(sys, e)
y = [item for sublist in y for item in sublist]
print('Experimental mean of y(t) is =', np.mean(y))
print('Experimental variance of y(t) is =', np.var(y))
print('First five values of y(t) using dlsim command are {}'.format(y[0:5]))

# %%============================================================================================================
#3. Write the AR(2) process in question 1, as multiple regression model and using the least square estimate (LSE),
# estimate the true parameters 𝑎1 and 𝑎2 (0.5, 0.2).
# Display the estimated parameters values at the console.
# What is the effect of additional samples on the accuracy of the estimate?
# Justify the answer by running your code for 5000 and 10000 data samples.
# %%------------------------------------------------------------------------------------------------------------
# %%============================================================================================================
#4. Generalized your code in the previous question, such when the code runs it asks a user the
# following questions:
# a. Enter number of samples :
# b. Enter the order # of the AR process:
# c. Enter the corresponding parameters of AR process :
    # Your code should simulate the AR process based on the entered information (a, b, c) and estimate the AR
    # parameters accordingly. The estimated parameters must be close to the entered numbers in part c.
    # Display the estimated parameters and the true values at the console.
# d. Increase the number of samples to 5000 and display the estimated parameters.
# e. Increase the number of samples to 10000 and display the estimated parameters.
# f. Write down your observation on the effect of the additional samples on the accuracy of
# the estimation.
# %%------------------------------------------------------------------------------------------------------------

def AR(T, order, params):
    mean = 0
    std = np.sqrt(1)
    np.random.seed(10)
    e = np.random.normal(mean, std, size=T)
    # print('The mean of WN is ', np.mean(e), 'The variance of the WN is ', np.var(e))
    num = [1, 0, 0]
    den = [1]
    den.extend(params)
    sys = (num, den, 1)
    _, y = signal.dlsim(sys, e)
    y = [item for sublist in y for item in sublist]
    T_new = len(y) - order - 1
    Y = pd.DataFrame(y[order:len(y)])
    X = np.zeros((T_new + 1, order))
    y_l = list(y)
    k = 1
    for j in range(order):
        for i in range(T_new + 1):
            X[i][j] = y_l[order+i-k]
        k += 1
    X = -1 * pd.DataFrame(X)
    x_transpose = X.transpose()
    coeff = np.linalg.inv(np.array(x_transpose.dot(X))).dot(x_transpose.dot(Y))
    coeff = [round(item, 3) for sublist in coeff for item in sublist]
    print('Actual Coefficients for AR({}) with {} samples are {}'.format(order, T, params))
    print('Estimated Coefficients for AR({}) with {} samples are {}'.format(order, T, coeff))
    return coeff, X, Y

T = int(input('Enter the number of samples: '))
order = int(input('Enter the order # of the AR process: '))
print("You need to enter {} coefficients since the order of the AR process is {}".format(order, order))
params = []
for i in range(1, order+1):
    params.append(float(input('Enter the coefficient {} of AR process :'.format(i))))

coeff, X, Y = AR(T, order, params)
coeff, X, Y = AR(T=1000, order=2, params=[-0.5, -0.2])
coeff, X, Y = AR(T=5000, order=2, params=[-0.5, -0.2])
coeff, X, Y = AR(T=10000, order=2, params=[-0.5, -0.2])

# %%============================================================================================================
#5. Let consider an MA(2) process as 𝑦(𝑡) = 𝑒(𝑡) + 0.5𝑒(𝑡 − 1) + 0.2𝑒(𝑡 − 2) Where 𝑒(𝑡) is a WN (1,2).
# a. Find the theoretical mean and variance of y(t). (no need to use python).
# b. Using python, create a for loop simulate above process for 1000 samples. Assume all initial conditions to be zero.
# c. Plot the y(t) with respect to number of samples.
# d. Using the python code, developed in previous labs, calculate autocorrelations for 20 lags
# and plot them versus number of lags. Write down your observation about the ACF of above process.
# e. Increase the data samples to 10000 and then 100000. Observe the ACF. Write down your
# observation. There is a difference between the ACF of an AR process and MA process.
# What is the main difference?
# f. Display the first 5 values of y(t) at the console.
# g. Apply the ADF-test and check if this is a stationary process. Explain your answer.
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "MA PROCESS" + 20 * "-")
def MA(T):
    mean = 0
    std = np.sqrt(1)
    np.random.seed(10)
    e = np.random.normal(mean, std, size=T)
    # print('The mean of WN is ', np.mean(e), 'The variance of the WN is ', np.var(e))
    y = np.zeros(len(e))
    for t in range(len(e)):
        if t == 0:
            y[t] = e[t]
        elif t == 1:
            y[t] = e[t] + 0.5*e[t-1]
        else:
            y[t] = e[t] + 0.5*e[t-1] + 0.2*e[t-2]
    return y

y = MA(T=1000)
print('The mean of MA(2) is ', np.mean(y), 'The variance of the MA(2) is ', np.var(y))

plt.figure()
plt.plot(y)
plt.xlabel('Number of Samples')
plt.ylabel('y(t)')
plt.title('y(t) vs number of samples')
plt.show()

k = 20
result = cal_auto_corr(y, k)
plt.figure()
plt.stem(range(-(k-1),k), result, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot of MA for 1000 samples')
plt.show()

y_10000 = MA(T=10000)
y_100000 = MA(T=100000)

result = cal_auto_corr(y_10000, k)
plt.figure()
plt.stem(range(-(k-1),k), result, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot of MA for 10000 Samples')
plt.show()

result = cal_auto_corr(y_100000, k)
plt.figure()
plt.stem(range(-(k-1),k), result, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot of MA for 100000 samples')
plt.show()

print('First five values of y(t) are {}'.format(y[0:5]))
ADFcal(y)

# %%============================================================================================================
#6. Using the “scipy” python package and “dlsim” command, simulate the MA(2) process in question
    # a. Display the first 5 values of y(t) at the console.
    # b. Show that your answer to the previous part is identical to the answer in part f of the previous question.
# %%------------------------------------------------------------------------------------------------------------
T = 1000
np.random.seed(10)
e = np.random.normal(mean, std, size=T)
num = [1, 0.5, 0.2]
den = [1, 0, 0]
sys = (num, den, 1)
_, y =signal.dlsim(sys, e)
y = [item for sublist in y for item in sublist]
print('Experimental mean of y(t) is =', np.mean(y))
print('Experimental variance of y(t) is =', np.var(y))
print('First five values of y(t) using dlsim command are {}'.format(y[0:5]))

plt.figure()
plt.plot(y)
plt.xlabel('Number of Samples')
plt.ylabel('y(t)')
plt.title('y(t) vs number of samples')
plt.show()
