import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
# %%============================================================================================================
# 1: Load the time series data called tute1.
# %%------------------------------------------------------------------------------------------------------------
q_sales = pd.read_csv("tute1.csv")
print("Shape of Tute1:",q_sales.shape)
print('\n')
# %%============================================================================================================
# 2: This date relates to the quarterly sales for a small company over period 1981-2005.
# %%------------------------------------------------------------------------------------------------------------
# print(q_sales.columns)
for i in q_sales.columns:
        print(i)
print('\n')
date_rng = pd.date_range(start='3/1/1981', end='3/1/2006', freq='Q')
print(date_rng)
# %%============================================================================================================
# 3: Sales contains the quarterly sales, AdBudget is the advertisement budget and GPD is the gross domestic
# product for a small company.
# %%------------------------------------------------------------------------------------------------------------
q_sales.rename( columns={'Unnamed: 0':'Q-SalesDate'}, inplace=True )
print(q_sales.head(5))
print('\n')
print(q_sales.tail(5))
print('\n')
print("DataType of Tute1:",q_sales.dtypes)
# %%============================================================================================================
# 4: Plot Sales, AdBudget and GPD versus time step.
# %%------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(16,10))
plt.plot(date_rng, q_sales['Sales'], 'b-', label = 'Sales')
plt.plot(date_rng, q_sales['AdBudget'], 'r-', label = 'AdBudget')
plt.plot(date_rng, q_sales['GDP'], 'y-', label = 'GDP')
plt.xlabel('Sales - Period (1981-2005)')
plt.ylabel('Quarterly Sales, Advertisement Budget and GDP in $')
plt.title('Time Series plot of Sales, Advertisement Budget & GDP of small Company')
plt.legend(loc='best')
plt.show()

# %%============================================================================================================
# 5: Find the time series statistics (average, variance and standard deviation) of Sales, AdBudget and GPD.
# %%------------------------------------------------------------------------------------------------------------
# print("MEAN")
m_sales = np.round(np.mean(q_sales['Sales']))
#print(m_sales)
m_AdBudget = np.round(np.mean(q_sales['AdBudget']))
#print(m_AdBudget)
m_GDP = np.round(np.mean(q_sales['GDP']))
#print(m_GDP)
# print("\n")

# print("VARIANCE")
var_sales = np.round(np.var(q_sales['Sales']))
#print(var_sales)
var_adbudget = np.round(np.var(q_sales['AdBudget']))
#print(var_adbudget)
var_gdp = np.round(np.var(q_sales['GDP']))
#print(var_gdp)
# print("\n")

# print("STANDARD DEVIATION")
std_sales = np.round(np.std(q_sales['Sales']))
#print(std_sales)
std_adbudget = np.round(np.std(q_sales['AdBudget']))
#print(std_adbudget)
std_gdp = np.round(np.std(q_sales['GDP']))
#print(std_gdp)

# %%============================================================================================================
# 6: Display the Average, variance, and standard deviation as follow:
# a. The Sales mean is : -------- and the variance is : -------- with standard deviation : -------
# b. The AdBudget mean is : -------- and the variance is : -------- with standard deviation : -----
# c. The GDP mean is : -------- and the variance is : -------- with standard deviation : -------
# %%------------------------------------------------------------------------------------------------------------
print("The Sales mean is : {0} and the variance is : {1} with standard deviation :{2}"
      .format(m_sales,var_sales,std_sales))
print("The AdBudget mean is : {0} and the variance is : {1} with standard deviation :{2}"
      .format(m_AdBudget,var_adbudget,std_adbudget))
print("The GDP mean is : {0} and the variance is : {1} with standard deviation : {2}"
      .format(m_GDP,var_gdp,std_gdp))

# %%============================================================================================================
# 7: Prove that the Sales, AdBudget and GDP in this time series dataset is stationary.
# Hint: To show a process is stationary, you need to show that data statistics is not changing by time.
# You need to create 100 sub-sequences from the original sequence and save the average and variance of each
# sub-sequence.
# Plot all means and variances and show that the means and variances are almost constant. To create sub-sequences,
# start with a sequence with the first sales data and find the mean.
# Then create another sub-sequence by adding the second sales date to the first sub-sequence,
# then find the corresponding mean. Repeat this process till you added the last sales date to the last sub-sequence and
# find the average. Repeat the same procedures for variances.
# Hint: Create a loop for the length of the dataset and use the following command to bring new data sample at
# each iteration: pd.read_csv('tute1.csv').head(i)# where is the number of samples
# %%------------------------------------------------------------------------------------------------------------
Sales_avg, AdBudget_avg, GDP_avg, Sales_var, AdBudget_var, GDP_var = [], [], [], [], [], []
for i in range(1,101):
    k = pd.read_csv('tute1.csv').head(i)
    Sales_avg.append(np.round(np.mean(k['Sales']),1))
    AdBudget_avg.append(np.round(np.mean(k['AdBudget']),1))
    GDP_avg.append(np.round(np.mean(k['GDP']),1))
    Sales_var.append(np.round(np.var(k['Sales']), 1))
    AdBudget_var.append(np.round(np.var(k['AdBudget']), 1))
    GDP_var.append(np.round(np.var(k['GDP']), 1))

plt.figure(figsize=(16,10))
plt.plot(date_rng, Sales_avg,'b-',label = 'Sales average')
plt.plot(date_rng, AdBudget_avg,'r-',label = 'AdBudget average')
plt.plot(date_rng, GDP_avg, 'y-',label = 'GDP average')
plt.xlabel("Sales - Period (1981-2005)")
plt.ylabel("Average of Sales,Adbudget & GDP  in $")
plt.title('Time series plot of Average Sales data')
plt.legend(loc='best')
plt.show()
# %%============================================================================================================
# 8:Plot all average and variance.
# Write down your observation about if this time series date is stationary or not? Why?
# %%------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(16,10))
plt.plot(date_rng, Sales_var,'b-',label = 'Sales variance')
plt.plot(date_rng, AdBudget_var,'r-',label = 'AdBudget variance')
plt.plot(date_rng, GDP_var, 'y-',label = 'GDP variance')
plt.xlabel("Sales - Period (1981-2005)")
plt.ylabel("Variance of Sales, AdBudget & GDP in $")
plt.title('Time series plot of Sales variance data')
plt.legend(loc='best')
plt.show()
# %%============================================================================================================
# 9: Perform an ADF-test to check if the Sales, AdBudget and GDP stationary or not (confidence interval 95% or above).
# Does your answer for this question reinforce your observations in the previous step?
# %%------------------------------------------------------------------------------------------------------------
def ADFcal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" %result[1])
    print("Critical values :")
    for key,value in result[4].items():
        print("\t%s: %.3f" % (key, value))
    print()
print()
print("#### ADF test on Sales ####")
ADFcal(q_sales['Sales'])

print("#### ADF test on AdBudget ####")
ADFcal(q_sales['AdBudget'])

print("#### ADF test on GDP ####")
ADFcal(q_sales['GDP'])

# %%============================================================================================================
# 10: Add an appropriate x-label, y-label, legend, and title to each graph.
# %%------------------------------------------------------------------------------------------------------------



