import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set_style('darkgrid')

df = pd.read_csv('AirPassengers.csv', index_col='Month', parse_dates=True)
Year = pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS')
y = df['#Passengers'].astype(float)


def plot_ma(y, k, trend, detrend, ma_order, folding_order):
    plt.figure(figsize=(16,10))
    plt.plot(np.array(Year[:50]), np.array(y[:50]), label='original')
    if ma_order%2 != 0:
        plt.plot(np.array(Year[k:50]), np.array(trend[:50-k]), label='{}-MA'.format(ma_order))
        plt.title('Plot for {}-MA'.format(ma_order))
    else:
        plt.plot(np.array(Year[k:50]), np.array(trend[:50 - k]), label='{}x{}-MA'.format(folding_order, ma_order))
        plt.title('Plot for {}x{}-MA'.format(folding_order, ma_order))
    plt.plot(np.array(Year[k:50]), np.array(detrend[:50-k]), label='detrended')
    plt.xlabel('Year')
    plt.ylabel('# of Passengers')
    plt.legend()
    plt.show()


def ADF_Cal(x):
    result = adfuller(x)
    print('ADF Statistic: %f' %result[0])
    print('p-value: %f' %result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


m = int(input('Enter the order of moving average: '))
while m <= 2:
    print('Sorry!! order 1,2 will not be accepted')
    m = int(input('Enter the order of moving average: '))
if m % 2 == 0:
    n = int(input('Enter the order of moving average: '))
    while n < 2 or n % 2 != 0 or n >= m:
        print('Sorry!! folding order should not be odd or not less than 2')
        n = int(input('Enter the order of moving average: '))


def cal_moving_average(col, ma_order, folding_order):
    ma = []
    k = int(np.ceil((ma_order - 1) / 2))
    for t in range(0, len(col) - ma_order + 1):
        temp = np.sum(y[t:ma_order + t])
        ma.append(temp / ma_order)

    if folding_order > len(ma):
        print("Invalid Folding order. Moving Average cannot be calculated if folding order is greater than the length of first moving average result")
    # passing folding order as zero for odd order of moving average
    elif folding_order != 0:
        k1 = int(np.ceil((ma_order - 1) / 2) + ((folding_order - 1) / 2))
        folding_ma = []
        for t in range(0, len(ma) - folding_order + 1):
            a = np.sum(y[t:folding_order + t])
            folding_ma.append(a / folding_order)
        print("Result of {}x{}-MA is: {}".format(folding_order, ma_order, folding_ma))
        detrended = np.divide(list(y.iloc[k1:-k1]), folding_ma)
        plot_ma(y, k1, ma, detrended, ma_order, folding_order)
        return detrended, folding_ma
    else:
        print("Result of {}-MA is: {}".format(ma_order, ma))
        detrended = np.divide(list(y.iloc[k:-k]), ma)
        plot_ma(y, k, ma, detrended, ma_order, folding_order)
        return detrended, ma


if m % 2 != 0:
    res_ma = cal_moving_average(col=y, ma_order=m, folding_order=0)
else:
    res_ma = cal_moving_average(col=y, ma_order=m, folding_order=n)

detrended_3, ma_3 = cal_moving_average(col=y, ma_order=3, folding_order=0)
detrended_5, ma_5 = cal_moving_average(col=y, ma_order=5, folding_order=0)
detrended_7, ma_7 = cal_moving_average(col=y, ma_order=7, folding_order=0)
detrended_9, ma_9 = cal_moving_average(col=y, ma_order=9, folding_order=0)

detrended_2x4, ma_2x4 = cal_moving_average(col=y, ma_order=4, folding_order=2)
detrended_2x6, ma_2x6 = cal_moving_average(col=y, ma_order=6, folding_order=2)
detrended_2x8, ma_2x8 = cal_moving_average(col=y, ma_order=8, folding_order=2)
detrended_2x10, ma_2x10 = cal_moving_average(col=y, ma_order=10, folding_order=2)

ADF_Cal(y)
ADF_Cal(detrended_3)

STL = STL(y)
res = STL.fit()
fig = res.plot()
# plt.fig(figsize=(16,10))
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

plt.figure(figsize=(16,10))
plt.plot(T, label='trend')
plt.plot(S, label='Seasonal')
plt.plot(R, label='residuals')
plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.title('Trend, Seasonality, Residual components using STL Decomposition')
plt.legend()
plt.show()

adjusted_seasonal = y-S
plt.figure(figsize=(16,10))
plt.plot(y, label='Original')
plt.plot(adjusted_seasonal, label='Seasonally Adjusted')
plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.title('Original vs Seasonally adjusted')
plt.legend()
plt.show()

# Measuring strength of trend and seasonality
F = np.max([0,1-np.var(np.array(R))/np.var(np.array(T+R))])
print('Strength of trend for Air Passengers dataset is', round(F,3))

FS = np.max([0, 1-np.var(np.array(R))/np.var(np.array(S+R))])
print('Strength of seasonality for Air Passengers dataset is', round(FS,3))
