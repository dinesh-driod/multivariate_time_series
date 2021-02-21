import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import warnings
warnings.filterwarnings("ignore")

# %%============================================================================================================
#1 . 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) = 𝑒(𝑡)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "𝑦(𝑡) − 0.5𝑦(𝑡 − 1) = 𝑒(𝑡)" + 20 * "-")
lags = 15
T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y = (1*(1+1))/((1-0.5))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))
    # an_sum = np.sum(an[i])
    # print("Sum of an[i] = ",an_sum)

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))
    # bn_sum = np.sum(bn[i])
    # print("Sum of bn[i] = ",bn_sum)

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y
# y = y + mean_y

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")
#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")

# %%------------------------------------------------------------------------------------------------------------
print(" Repeat for 5000 samples")
T1 = int(input("Enter the number of data samples:"))
y1 = arma_process.generate_sample(T1, scales) + mean_y

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y1, 15)
plot_acf(autocorrelation, "ACF plot for y")

# autocorrelation
ry = cal_auto_correlation(y1, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")
# %%------------------------------------------------------------------------------------------------------------
print(" Repeat for 10000 samples")
T2 = int(input("Enter the number of data samples:"))
y2 = arma_process.generate_sample(T2, scales) + mean_y

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y2, 15)
plot_acf(autocorrelation, "ACF plot for y")

# autocorrelation
ry = cal_auto_correlation(y2, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")
print()

# %%============================================================================================================
#2 . y(t) = e(t) + 0.5e(t-1)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "y(t) = e(t) + 0.5e(t-1)" + 20 * "-")
T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y2 = (1*(1+0.5))/((1))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))
    # an_sum = np.sum(an[i])
    # print("Sum of an[i] = ",an_sum)

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))
    # bn_sum = np.sum(bn[i])
    # print("Sum of bn[i] = ",bn_sum)

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y2

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")

#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")

# %%============================================================================================================
#3 . ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)" + 20 * "-")
T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y3 = (1*(1+0.5))/((1+0.5))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y3

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")

#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")

# %%============================================================================================================
#4 . ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)" + 20 * "-")
T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y4 = (1*(1))/((1+0.5+0.2))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y4

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")

#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")
# %%============================================================================================================
#5 . ARMA (2,1): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "ARMA (2,1): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1) " + 20 * "-")
T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y5 = (1*(1-0.5))/((1+0.5+0.2))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y5

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")

#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")

# %%============================================================================================================
#6 . ARMA (1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "ARMA (1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2) " + 20 * "-")

T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y6 = (1*(1+0.5-0.4))/((1+0.5))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y6

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")

#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")

# %%============================================================================================================
#7 .ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)" + 20 * "-")

T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y7 = (1*(1+0.5-0.4))/((1))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y7

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")

#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")

# %%============================================================================================================
#8. ARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)
# %%------------------------------------------------------------------------------------------------------------
print(20 * "-" + "ARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)" + 20 * "-")

T = int(input("Enter the number of data samples :"))
na = int(input("Enter AR order:"))
nb = int(input("Enter MA order:"))
mean = int(input("Enter Mean of white noise:"))
std = int(input("Enter variance of white noise:"))
scales = np.sqrt(std)
mean_y8 = (1*(1+0.5-0.4))/((1+0.5+0.2))

an = [0]*na
bn = [0]*nb

for i in range(na):
    an[i] = float(input("Enter coefficient of AR a{}".format(i+1)))

for i in range(nb):
    bn[i] = float(input("coefficient of MA b{}".format(i+1)))

max_order = max(na,nb)
num = [0]*(max_order+1)
den = [0]*(max_order+1)
for i in range(na+1):
    if i==0:
        den[i] = 1
    else:
        den[i] = an[i-1]

arparmas = np.array(an)
maparams = np.array(bn)

na = len(arparmas)
nb = len(maparams)

ar = np.r_[1, arparmas]
ma = np.r_[1, maparams]

arma_process = sm.tsa.ArmaProcess(ar,ma)
print("Is this stationary process: ", arma_process.isstationary)

y = arma_process.generate_sample(T, scales) + mean_y8

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(y, 15)
plot_acf(autocorrelation, "ACF plot for y")

#gpac
j = 7
k = 7
lags = j + k

# autocorrelation
ry = cal_auto_correlation(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print("GPAC Table for y:")
print(gpac_table.to_string())

# heatmap
plot_heatmap(gpac_table, "GPAC Table for y")



