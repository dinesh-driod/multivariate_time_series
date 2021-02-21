import os
import pandas as pd
from requests import get
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import STL
from functions import *
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()


# %%============================================================================================================
# DATA SET DESCRIPTION
# %%------------------------------------------------------------------------------------------------------------
'''
1: Objective:
To predict the Relative Humidity of a given point of time based on the all other attributes affecting the change  in RH
'''
# %%============================================================================================================
#                                           LOAD THE DATASET
# %%------------------------------------------------------------------------------------------------------------
print('\n')
print("loading dataset....")
if "AirQualityUCI" not in os.listdir():
    request = get('https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip')
    zip_file = ZipFile(BytesIO(request.content))
    zip_file.extractall()
print('\n')
df = pd.read_csv("AirQualityUCI.csv", sep = ';',infer_datetime_format=True)
print("Download complete....")

# %%============================================================================================================
#                                              PREPROCESSING
# %%------------------------------------------------------------------------------------------------------------
print('\n')
print(20 * "-" + "PREPROCESSING" + 20 * "-")
print(df.head(5))

# Removing last two Unnamed columns
df = df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1)
print('\n',df.info)

# Changing the datatype from object to float
print('\n',df.dtypes)
df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.').astype(float)
df['C6H6(GT)'] = df['C6H6(GT)'].str.replace(',','.').astype(float)
df['T'] = df['T'].str.replace(',', '.').astype(float)
df['RH'] = df['RH'].str.replace(',', '.').astype(float)
df['AH'] = df['AH'].str.replace(',', '.').astype(float)

# Dimension of the Dataset
print('\n')
print('Shape of the Dataset:',df.shape)
# %%------------------------------------------------------------------------------------------------------------

print('\n')
# Estimating null values
print(df.isnull().sum())

# Remving null values
null_data = df[df.isnull().any(axis=1)]
print('\n',null_data.head())
df= df.dropna()
# print(df.head)
print('Shape of the Dataset after null value removal',df.shape)

# Replacing -200 with nan
df = df.replace(-200,np.nan)
print(df.isnull().sum())

# Appending date and time
print(df.index)
df.loc[:,'Datetime'] = df['Date'] + ' ' + df['Time']
DateTime = []
for x in df['Datetime']:
    DateTime.append(datetime.strptime(x,'%d/%m/%Y %H.%M.%S'))
datetime = pd.Series(DateTime)
df.index = datetime
# print(df.head())
# print('AFTER',df.dtypes)
df = df.replace(-200, np.nan)
print(df.isnull().sum())
print(df.head)
# %%------------------------------------------------------------------------------------------------------------
# CREATING PROCESSED DATAFRAME
# SD = df['Date']
# ST = df['Time']
S0 = df['CO(GT)'].fillna(df['PT08.S1(CO)'].mean())
S1 = df['PT08.S1(CO)'].fillna(df['PT08.S1(CO)'].mean())
S2 = df['NMHC(GT)'].fillna(df['NMHC(GT)'].mean())
S3 = df['C6H6(GT)'].fillna(df['C6H6(GT)'].mean())
S4 = df['PT08.S2(NMHC)'].fillna(df['PT08.S1(CO)'].mean())
S5 = df['NOx(GT)'].fillna(df['NOx(GT)'].mean())
S6 = df['PT08.S3(NOx)'].fillna(df['PT08.S1(CO)'].mean())
S7 = df['NO2(GT)'].fillna(df['NO2(GT)'].mean())
S8 = df['PT08.S4(NO2)'].fillna(df['PT08.S1(CO)'].mean())
S9 = df['PT08.S5(O3)'].fillna(df['PT08.S1(CO)'].mean())
S10 = df['T'].fillna(df['T'].mean())
S11 = df['RH'].fillna(df['RH'].mean())
S12 = df['AH'].fillna(df['AH'].mean())

print('AFTER MEAN\n',df.isnull().sum())
print('\n')
df = pd.DataFrame({'CO(GT)':S0,'PT08.S1(CO)':S1,'NMHC(GT)':S2, 'C6H6(GT)':S3, 'PT08.S2(NMHC)':S4, 'NOx(GT)':S5,
                   'PT08.S3(NOx)':S6, 'NO2(GT)':S7,  'PT08.S4(NO2)':S8, 'PT08.S5(O3)':S9, 'T':S10, 'RH':S11, 'AH':S12 })
print("THE CLEANED DATASET AFTER PREPROCESSING:\n", df)

# df.index = datetime
# df.to_csv("AQI.csv")

# print('CLEANDED DATASET:\n')
# print(df.isnull().sum())
# print('CLEANDED DATASET:\n',df.head)
# %%============================================================================================================
#                                              HOMEWORK 9 - Order estimation using GPAC Table
# %%------------------------------------------------------------------------------------------------------------

# split into train and test(20%) dataset
train, test = split_df_train_test(df, 0.2)
print()
print("The dimension of train data is:")
print(train.shape)
# dimension of test data
print()
print("The dimension of test data is:")
print(test.shape)

# Created Dataframe for Dependent variable and time
df_rh = pd.DataFrame({'RH':S11})
# df.to_csv("AirQuality_processed_rh.csv")
print('\n',df_rh.head())

# DEPENDENT VARIABLE V/S TIME
plt.figure(figsize=(16,10))
plt.plot(df_rh, 'b-', label = 'RH')
plt.xlabel('Time | March 2004- February 2005')
plt.ylabel('RH')
plt.title('Time Series plot of Relative humidity')
plt.legend(loc='best')
plt.show()

# AUTO CORRELATION USING BULTIN FUNCTION
plot_acf(df_rh)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF using statsmodel')
plt.show()

# AUTO CORRELATION USING CREATED FUNCTION
y = df['RH']
k = 20
acfcal = auto_corr_cal(y,k)

# acfcal = cal_auto_corr(y,k)
acfplotvals = acfcal[::-1] + acfcal[1:]
plt.figure(figsize=(16,10))
plt.stem(range(-(k - 1), k), acfplotvals)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF using custom function')
plt.show()

corr = df.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.figure(figsize=(15,10))
plt.show()

print('\n')
test_result = adfuller(df['RH'])
adfuller_test(df['RH'])

j = 8
k = 8
lags = j + k

y_mean = np.mean(train['RH'])
y = np.subtract(y_mean, df['RH'])
actual_output = np.subtract(y_mean, test['RH'])
ry = auto_corr_cal(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print()
print("GPAC Table:")
print(gpac_table.to_string())
print()
plot_heatmap(gpac_table, "GPAC Table for RH")









