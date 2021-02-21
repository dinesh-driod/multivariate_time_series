#Autocorrelation Function

import numpy as np

def auto_corr(y, k):
    T = len(y)
    y_mean = np.mean(y)
    res_num = 0
    res_den = 0
    for t in range(k, T):
        res_num += (y[t] - y_mean) * (y[t-k] - y_mean)

    for t in range(0, T):
        res_den += (y[t] - y_mean)**2

    result = res_num/res_den
    return result

def cal_auto_corr(y, k):
    res = []
    res1 = []
    for t in range(0, k):
        result = auto_corr(y, t)
        res.append(result)
    for t in range(k-1, 0, -1):
        res1.append(res[t])
    res1.extend(res)
    return res1
