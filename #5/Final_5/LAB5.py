import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
import seaborn as sns

from Autocorrelation import cal_auto_corr
from Pearson_Correlation_Coefficient import correlation_coefficent_cal

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
# %%============================================================================================================
#Using Pandas library load the time series data from the BB.
# Split the dataset into training setand test set.
# Use 80% for training and 20% for testing. Display the training set and testing set array as follow:
# Hint: This can be done using the following library in python.
# “from sklearn.model_selection import train_test_split” . Make sure the “Shuffle=False”
# %%------------------------------------------------------------------------------------------------------------
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price','normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size',
         'bore', 'stroke','compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']]

features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()

features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)

# %%============================================================================================================
#2: Plot the correlation matrix using the seaborn package and heatmap function.
# %%============================================================================================================
corr = df.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.figure(figsize=(15,10))
plt.show()

# %%============================================================================================================
#3: Using python, construct matrix X and Y using x-train and y-train dataset and estimate the
# regression model unknown coefficients using the Normal equation (LSE method, above
# equation). Display the unknown coefficients on the console. Note: You are not allowed to use
# OLS for this part.
# %%------------------------------------------------------------------------------------------------------------
def plot_fun(train, test, predicted, title):
    plt.figure(figsize=(15,10))
    plt.plot(range(0, len(train)), train, label='Training set')
    plt.plot(range(len(train), len(train)+len(test)), test, label='Testing set')
    plt.plot(range(len(train), len(train)+ len(predicted)), predicted, label='prediction values')
    plt.xlabel('')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()

x_transpose = np.transpose(x_train)
beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_transpose, x_train)), x_transpose), y_train)
print('\n')
print('Unknown Coefficients:',beta_hat)

y_hat = np.matmul(x_train, beta_hat)
eps = y_train - y_hat
y_test_hat = np.matmul(x_test, beta_hat)
plot_fun(y_train, y_test, y_test_hat, 'LSE Method')

# %%============================================================================================================
#4: Using python, statsmodels package and OLS function, find the unknown coefficients.
# Compare the results with the step 3. Display the result on the console.
# Are the unknown coefficient calculated using step 3 and step 4 the same?
# %%------------------------------------------------------------------------------------------------------------
print('\n')
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# %%============================================================================================================
# 5: Perform a prediction for the length of test-set and plot the train, test and predicted values in
# one graph. Add appropriate x-label, y-label, title, and legend to your graph.
# %%------------------------------------------------------------------------------------------------------------
y_hat_OLS = model.predict(x_train)
eps_OLS = y_train - y_hat_OLS
y_test_hat_OLS = model.predict(x_test)
plot_fun(y_train, y_test, y_test_hat_OLS, 'OLS Method')

prediction_error = y_train - y_hat_OLS
forecast_error = y_test - y_test_hat_OLS
lags = 20
prediction_error_acf = cal_auto_corr(prediction_error, lags)
forecast_error_acf = cal_auto_corr(forecast_error, lags)

# %%============================================================================================================
# 6: Calculate the prediction errors and plot the ACF of prediction errors.
# Write down your observation
# %%------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.stem(range(-(lags-1),lags), prediction_error_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Prediction Error')
plt.show()


# %%============================================================================================================
# 7: Calculate the forecast errors and plot the ACF of forecast errors.
# Write down your observation.
# %%------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.stem(range(-(lags-1),lags), forecast_error_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Forecast Error')
plt.show()

# %%============================================================================================================
#8: Calculate the estimated variance of the prediction errors and the forecast errors. Compare the results?
# What is your observation? Hint: Use need to the following equation to estimate the variance:
# %%------------------------------------------------------------------------------------------------------------
print('\n')
T = len(x_train)
K = len(x_train[0, 1:])
pred_var = (1/(T-K-1)) * (np.sum((prediction_error)**2))
print("variance of prediction error: ", pred_var)

T = len(x_test)
K = len(x_test[0, 1:])
forecast_var = (1/(T-K-1)) * (np.sum((forecast_error)**2))
print("variance of forecast error: ", forecast_var)
print('\n')

# %%============================================================================================================
#9: Plot the scatter plot between y-test and 𝑦̂𝑡+ℎ and display the correlation coefficient between them on the title.
# Justify the accuracy of this predictor by observing the correlation coefficient between y-test and 𝑦̂𝑡.
# %%------------------------------------------------------------------------------------------------------------
corr_coeff = round(correlation_coefficent_cal(y_test, y_test_hat_OLS),2)
print('correlation coefficient between y-test and 𝑦̂𝑡:',corr_coeff)
print('\n')

plt.figure(figsize=(15, 10))
plt.scatter(y_test, y_test_hat_OLS, c='green', alpha=1, label='y_test vs y_test_hat_OLS')
plt.xlabel('y_test')
plt.ylabel('y_test_hat_OLS')
plt.title("Scatter plot of y_test vs y_hat_test with correlation coefficient of {}".format(corr_coeff))
plt.legend()
plt.show()

# %%============================================================================================================
#10: Using a stepwise regression, try to reduce the feature space dimension.
# You need to use the AIC, BIC and Adjusted R2 as a predictive accuracy for your analysis.
# If your analysis recommends an elimination, which feature(s) would you eliminate?
# You can use the backward or forward or hybrid stepwise regression for feature selection.
#11: Perform a complete t-test and F-test analysis on the final model and write down your observations
# %%------------------------------------------------------------------------------------------------------------
# Removing bore
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price','normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size',
         'stroke','compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']]

features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

#Removing normalized losses
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'stroke',
         'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']]
features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

#Removing curb-weight
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price', 'wheel-base', 'length', 'width', 'height', 'engine-size', 'stroke',
         'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']]
features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

#Removing length
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price', 'wheel-base', 'width', 'height', 'engine-size', 'stroke','compression-ratio',
         'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']]
features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

#Removing height
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price', 'wheel-base', 'width', 'engine-size', 'stroke','compression-ratio', 'horsepower', 'peak-rpm',
         'city-mpg','highway-mpg']]
features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

#Removing highway-mpg
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price', 'wheel-base', 'width', 'engine-size', 'stroke','compression-ratio', 'horsepower', 'peak-rpm',
         'city-mpg']]
features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

#Removing city-mpg
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price', 'wheel-base', 'width', 'engine-size', 'stroke','compression-ratio', 'horsepower', 'peak-rpm']]
features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

#Removing width
df = pd.read_csv('autos.clean_corr.csv')
df = df[['price', 'wheel-base', 'engine-size', 'stroke','compression-ratio', 'horsepower', 'peak-rpm']]
features = df.drop(columns='price').to_numpy()
target = df['price'].to_numpy()
features = sm.add_constant(features)
x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())
