import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Autocorrelation import cal_auto_corr

df = pd.read_csv('tute1.csv', header=0)
df['Date'] = pd.date_range(start='1981-3-1', end='2006-3-1', freq='Q')
y = df['Sales']
y.index = df.Date
N = 100


def calc_coeff(y, order):
    T_new = len(y) - order - 1
    Y = np.array(y[order:len(y)])
    X = np.zeros((T_new + 1, order))
    y_l = list(y)
    k = 1
    for j in range(order):
        for i in range(T_new + 1):
            X[i][j] = y_l[order + i - k]
        k += 1
    X = -1 * pd.DataFrame(X)
    X = np.array(X)
    x_transpose = X.T
    coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_transpose, X)), x_transpose), Y)
    print('Coefficients for AR({}) are {}'.format(order, coeff))
    return coeff, X, Y


def MSE(error):
    return np.mean(error ** 2)


def cal_Q_value(residual_error, residual_error_acf, lags):
    k = len(residual_error)
    return k * np.sum(np.array(residual_error_acf[lags:]) ** 2)


def plot_acf(residual_error_acf, lags, title):
    plt.figure()
    plt.stem(range(-(lags - 1), lags), residual_error_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for {}'.format(title))
    plt.show()


def histogram_plot(residual_error, title):
    plt.figure()
    plt.hist(residual_error)
    plt.title('Histogram of {}'.format(title))
    plt.xlabel('values of {}'.format(title))
    plt.ylabel('Frequency')
    plt.show()


def plot_truevsestimated(df, order, Y, y_hat, title):
    plt.figure()
    plt.plot(df['Date'][order:], y_hat, label='Estimated Sales')
    plt.plot(df['Date'][order:], Y, label='Actual Sales')
    plt.title('Actual vs Estimated Sales for {}'.format(title))
    plt.xlabel('Time (Quarterly)')
    plt.ylabel('Sales')
    plt.legend(loc='upper left')
    plt.show()


def AR_model(df, y, order):
    result = []
    lags=20
    coeff, X, Y = calc_coeff(y, order)
    y_hat = np.matmul(X, coeff)
    residual_error = np.subtract(Y, y_hat)
    result.append(MSE(residual_error))
    print("MSE of the residual error for AR({}) model: {}".format(order, MSE(residual_error)))
    residual_error_acf = cal_auto_corr(residual_error, lags)
    Q_value = cal_Q_value(residual_error, residual_error_acf, lags)
    result.append(Q_value)
    print('Q value of residual errors for AR({}) model: {}'.format(order, Q_value))
    title = 'residual error (AR({}) model)'.format(order)
    plot_acf(residual_error_acf, lags, title=title)
    histogram_plot(residual_error, title)
    result.append(np.mean(residual_error))
    result.append(np.var(residual_error))
    print('Mean of residual errors for AR({}) model: {}'.format(order, np.mean(residual_error)))
    print('Variance of residual errors for AR({}) model: {}'.format(order, np.var(residual_error)))
    plot_truevsestimated(df, order, Y, y_hat, title='AR({}) Model'.format(order))
    return result, coeff

results = []
coefficients = {}
col_names = []
for i in range(1,6):
    result, coeff = AR_model(df, y, order=i)
    results.append(result)
    coefficients[i] = coeff
    col_names.append('AR({})'.format(i))

results_df = pd.DataFrame(data=results)
results_df = results_df.transpose()
results_df.columns = col_names
results_df.index = ['MSE', 'Q-value', 'Mean', 'Variance']
print(results_df.head())

# AR(4) is the best model
print('Best order number of AR model is: {} and their coefficients are: {}'. format(4, coefficients[4]))
