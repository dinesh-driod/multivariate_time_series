import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#Using the Python program and “pandas”,  “matplotlib.pyplot” and “numpy” library perform the following tasks:
# %%============================================================================================================
# 1: Write a python function called “ correlation_coefficent_cal(x,y)”  that implement the correlation coefficient.
# The formula for correlation coefficient is given below.
# The function should be written in a general form than can work for any dataset x and dataset y.
# The return value for this function is r.
# %%------------------------------------------------------------------------------------------------------------
def correlation_Coefficient_cal(X, Y):
    n = len(X)
    sum_X = 0
    sum_Y = 0
    sum_XY = 0
    squareSum_X = 0
    squareSum_Y = 0
    i = 0
    while i < n:
        sum_X = sum_X + X[i]
        sum_Y = sum_Y + Y[i]
        sum_XY = sum_XY + X[i] * Y[i]

        squareSum_X = squareSum_X + X[i] * X[i]
        squareSum_Y = squareSum_Y + Y[i] * Y[i]

        i = i + 1

    r = (float)(n * sum_XY - sum_X * sum_Y) / \
        (float)(math.sqrt((n * squareSum_X -sum_X * sum_X) * (n * squareSum_Y -sum_Y * sum_Y)))

    return r
print('\n')
# %%============================================================================================================
# 2: Test the “ correlation_coefficent_cal(x,y)” function with the following simple dataset.
# The x and y here are dummy variable and should be replaced by any other dataset.
# X = [1, 2, 3, 4, 5]
# Y = [1, 2, 3, 4, 5]
# Z = [-1, -2, -3, -4,-5]
# G = [1,1,0,-1,-1,0,1]
# H = [0,1,1,1,-1,-1,-1]
# %%------------------------------------------------------------------------------------------------------------
#Replacing the dummy variable to below variables
# Input Driver function
X = [15, 18, 21, 24, 27]
Y = [25, 25, 27, 31, 32]

print('Correlation Coeffiecent for X & Y:{0:.6f}'.format(correlation_Coefficient_cal(X, Y)))
print('\n')
# %%============================================================================================================
# a.Plot the scatter plot between X, Y
# %%------------------------------------------------------------------------------------------------------------
X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]

plt.xlabel("X_DATASET")
plt.ylabel("Y_DATASET")
plt.scatter(X, Y)
plt.show()
# %%============================================================================================================
# b.Plot the scatter plot between X, Z
# %%------------------------------------------------------------------------------------------------------------
X = [1, 2, 3, 4, 5]
Z = [-1, -2, -3, -4,-5]

plt.xlabel("X_DATASET")
plt.ylabel("Z_DATASET")
plt.scatter(X, Z)
plt.show()
# %%============================================================================================================
# c.Plot the scatter plot between G, H
# %%------------------------------------------------------------------------------------------------------------
G = [1,1,0,-1,-1,0,1]
H = [0,1,1,1,-1,-1,-1]

plt.xlabel("G_DATASET")
plt.ylabel("H_DATASET")
plt.scatter(G, H)
plt.show()
# %%============================================================================================================
# d.Without using Python program, implement the above formula to derive the r_xy, r_xz, r_gh.
# You should NOT use computer to answer this section.
# You need to show all your work for this section on the paper.
# %%------------------------------------------------------------------------------------------------------------
'''
See LAB2 report for manual calculations.
'''

# %%============================================================================================================
#e.Calculate r_xy , r_xz  and r_gh using the written python function “correlation_coefficent_cal(x,y)”.
# %%------------------------------------------------------------------------------------------------------------
X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]
r_xy = (correlation_Coefficient_cal(X, Y))


X = [1, 2, 3, 4, 5]
Z = [-1, -2, -3, -4,-5]
r_xz = (correlation_Coefficient_cal(X, Z))

G = [1,1,0,-1,-1,0,1]
H = [0,1,1,1,-1,-1,-1]
r_gh = (correlation_Coefficient_cal(G, H))

# %%============================================================================================================
#f.Compare the answer in section d and e. Any difference in value?
# %%------------------------------------------------------------------------------------------------------------
'''
Manual calculations matches with python calculations.
There is no difference in terms of manual and python calculations.
'''

# %%============================================================================================================
#g.	Display the message as:
#i.	The correlation coefficient between x and y is ______
#ii.	The correlation coefficient between x and z is ______
#iii.	The correlation coefficient between g and h is ______
# %%------------------------------------------------------------------------------------------------------------

print('The correlation coefficient between x and y is:{0:.6f}'.format(r_xy))
print('\n')
print('The correlation coefficient between x and z is:{0:.6f}'.format(r_xz))
print('\n')
print('The correlation coefficient between g and h is:{0:.6f}'.format(r_gh))
print('\n')
# %%============================================================================================================
#h.	Add an appropriate x-axis label and y-axis label to all your scatter graphs.
# %%------------------------------------------------------------------------------------------------------------
'''
Added appropriate x-axis label, y-axis label with titles for above scatter plots.

'''
# %%============================================================================================================
#i.	Include the r_xy , r_xz, r_gh as a variable on the scatter plot title in part a and part b.
# The code should be written in a way that the r value changes on the figure title automatically.
# Hint: You can use the following command:
#plt.title("Scatter plot of X and Y with r ={}".format(r_xy))
# %%------------------------------------------------------------------------------------------------------------
print("Scatter Plot for X and Y Dataset")
plt.xlabel("X_DATASET")
plt.ylabel("Y_DATASET")
plt.title("Scatter plot of X and Y with r ={}".format(r_xy))
plt.scatter(X, Y)
plt.show()
print('\n')
print("Scatter Plot for X and Z Dataset")
plt.xlabel("X_DATASET")
plt.ylabel("Z_DATASET")
plt.title("Scatter plot of X and Z with r ={}".format(r_xz))
plt.scatter(X, Z)
plt.show()
print('\n')
print("Scatter Plot for G and H Dataset")
plt.xlabel("G_DATASET")
plt.ylabel("H_DATASET")
plt.title("Scatter plot of G and H with r ={}".format(r_gh))
plt.scatter(G, H)
plt.show()

# %%============================================================================================================
#j.	Does the calculated r_xy, r_xz and r_gh make sense with respect to the scatter plots? Explain why?
# %%------------------------------------------------------------------------------------------------------------
'''
Yes it does make sense. 
Scatter plots showing different levels of correlation are plotted where 
Positive values indicating a positive relationship between Y_Dataset v/s X_Dataset.
Negative values indicating a negative relationship between Z_Dataset v/s X_Dataset.
Zero signifies there is no linear relationship between H_Dataset v/s G_Dataset
'''