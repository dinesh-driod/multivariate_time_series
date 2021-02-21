#Homework 8 : Partial Correlation Coefficient

# Using the “numpy” and “pandas” library and the python program you wrote for the correlation
# coefficient before, perform the followings steps:

import pandas as pd
import numpy as np
from CorrelationCoefficient import *
from scipy.stats import ttest_ind
print('\n')
# %%============================================================================================================
# 1: Load the “tute1.csv” dataset.
# Write a python program that calculate the correlation coefficient between Sales and AdBuget
# and display the following message on the console.
# “Correlation Coefficient between Sales and AdBugdet is ________”
# %%------------------------------------------------------------------------------------------------------------
df = pd.read_csv("tute1.csv")
s = df["Sales"]
ad = df["AdBudget"]
r_s_ad = (correlation_Coefficient_cal(s, ad))
print('The correlation coefficient between the Sales value and AdBudget is:{0:.2f}'.format(r_s_ad))
print('\n')

# %%============================================================================================================
# 2: Write a python program that calculate the correlation coefficient between AdBuget and GDP
# and display the following message on the console:
# “Correlation Coefficient between AdBugdet and GDP is ________”
# %%------------------------------------------------------------------------------------------------------------

g = df["GDP"]
r_ad_g = (correlation_Coefficient_cal(ad, g))
print('The correlation coefficient between the AdBudget and GDP is:{0:.2f}'.format(r_ad_g))
print('\n')

# %%============================================================================================================
# 3: Write a python program that calculate coefficient between Sales and GDP and display the
# following message on the console:
# “Correlation Coefficient between Sales and GDP is ________”
# %%------------------------------------------------------------------------------------------------------------
r_s_g = (correlation_Coefficient_cal(s, g))
print('The correlation coefficient between the Sales value and GDP is:{0:.2f}'.format(r_s_g))
print('\n')

# %%============================================================================================================
# 4: Using the hypothesis test (t-test) show whether the correlation coefficients in step 2, 3, and 4
# are statistically significant? Assume the level of confident to be 95% with two tails (𝛼 = 0.05 ).
# %%------------------------------------------------------------------------------------------------------------
n = df.shape[0]
degreef = n-2

t0 = r_ad_g*np.sqrt((n-2)/(1-r_ad_g**2))
print("Degree of Freedom =",degreef)
print("t0-value = ",t0)
print("Absolute value of test statistic did not exceed the critical t-value from the table. \n"
"The correlation coefficient (partial correlation)is statistically insignificant.")
print('\n')
t0 = r_s_g*np.sqrt((n-2)/(1-r_s_g**2))
print("t0-value = ",t0)
print("Absolute value of test statistic did not exceed the critical t-value from the table.\n"
"The correlation coefficient (partial correlation)is statistically insignificant.")
print('\n')

# %%============================================================================================================
# 5: Write a python program that calculate the partial correlation coefficient between Sales and AdBudegt.
# Using the hypothesis test, step 5, shows whether the derived coefficient is statistically significant.
# Write down your observation. Hint: Partial correlation coefficient
# between variable A and B with confounding variable C can be calculated as :
# %%------------------------------------------------------------------------------------------------------------
n = df.shape[0]
k = 1
degreef = n-2-k

r_sad_g = (r_s_ad - r_s_g*r_ad_g)/np.sqrt(1-r_s_g**2)*np.sqrt(1-r_ad_g**2)
print("partial coefficient between sales and adbudget for given gdp:",r_sad_g)
print("Degree of Freedom =",degreef)
t0 = r_sad_g*np.sqrt((n-2-k)/(1-r_sad_g**2))
print("t0-value = ",t0)
print("Absolute value of test statistic exceeded the critical t-value from the table.\n" 
"The correlation coefficient (partial correlation) is statistically significant.")
print('\n')
# %%============================================================================================================
# 6: Write a python program that calculate the partial correlation coefficient between Sales and
# GDP. Using the hypothesis test, shows whether this coefficient is statistically significant. Write
# down your observation. The t-value can be calculated as follow. The critical t-value can be found
# from the t-table.
# Where r is the partial correlation coefficient and n is the number of observations.
# %%------------------------------------------------------------------------------------------------------------
r_sg_ad = (r_s_g - r_s_ad*r_ad_g)/np.sqrt(1-r_s_g**2)*np.sqrt(1-r_ad_g**2)
print("partial coefficient between sales and gdp for given Adbudget:",r_sg_ad)

t0 = r_sg_ad*np.sqrt((n-2-k)/(1-r_sg_ad**2))
print("t0-value = ",t0)
print("Absolute value of test statistic did not exceed the critical t-value from the table.\n"
"The correlation coefficient (partial correlation)is statistically insignificant.")
print('\n')

# %%============================================================================================================
# 7:Write a python program that calculate the partial correlation coefficient between AdBudegt and GDP.
# Using the hypothesis test, shows whether this coefficient is statistically significant.
# Write down your observation.
# %%------------------------------------------------------------------------------------------------------------

r_adg_s = (r_ad_g - r_s_ad*r_s_g)/np.sqrt(1-r_s_ad**2)*np.sqrt(1-r_s_g**2)
print("partial coefficient between Adbudget and gdp for given sales:",r_adg_s)

t0 = r_adg_s*np.sqrt((n-2-k)/(1-r_adg_s**2))
print("t0-value = ",t0)
print("Absolute value of test statistic did not exceed the critical t-value from the table.\n"
"The correlation coefficient (partial correlation)is statistically insignificant.")
print('\n')

# %%============================================================================================================
# 8: Create a table and place all the results from step 2 through 8 inside the table.
# Compare the correlation coefficients and partial correlation coefficients for (Sales, AdBudget), (Sales, GDP)
# and (AdBudegt, GDP). Write down your observation.
# %%------------------------------------------------------------------------------------------------------------
'''
see Report
'''

# %%============================================================================================================
# 9: If you must drop one of the predictors (AdBudegt or GDP) which predictor do you pick for
# elimination? You need to justify your answer using the results above.
# %%------------------------------------------------------------------------------------------------------------
'''
In my opinion i would drop GDP since the correlation coefficient (partial correlation) is statistically significant for 
sales and adbudget for given gdp. see report
'''

