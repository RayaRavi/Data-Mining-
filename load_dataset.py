from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('/workspaces/codespaces-jupyter/DM LAB/Data.csv')
print("-------------THIS IS THE DATA IN THE FILE-------------\n")
print(data)
a=data.head()
b=data.tail()
print("-------------THIS IS THE DATA OF HEAD() IN DATA---------\n")
print(a)
print("-------------THIS IS THE DATA OF TAIL() IN DATA---------\n")
print(b)
x=data.iloc[:,:]
print("-----------THE VARIABLES IN THE GIVEN DATASET-----------------\n")
print(x)
x=data.iloc[:,0:3]
print("-----------THE VARIABLES IN THE GIVEN DATASET WITH SPECIFIC COLUMNS-----------------\n")
print(x)
x=data.iloc[:,:-1]
print("-----------THE INDEPENDENT VARIABLES IN THE GIVEN DATASET-----------------\n")
print(x)
y=data.iloc[:,-1:].values
print("-----------THE DEPENDENT VARIABLES IN THE GIVEN DATASET-----------------\n")
print(y)
print("-----------FOR MEAN OF AGE AND SALARY USING mean()\n")
a=data.Age.mean()#taken a for storing mean of Age attribute in Data.csv file
b=data.Salary.mean()#taken b for storing mean of Salary attribute in Data.csv file
print("----------MEAN OF AGE------------\n",b)
print("----------MEAN OF Salary----------\n",a)
#---------FOR DEALING WITH MISSING DATA----------------
print("--------------------FOR DEALING WITH MISSING DATA-----------------\n")
x=data.iloc[:,:-1].values
imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
#DEALING WITH MISSING DATA USING ATTRIBUTE NAMES
print("----------------DEALING WITH MISSING DATA USING ATTRIBUTE NAMES-------------------\n")
x=data.iloc[:,:-1].values
imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')
imputer = imputer.fit(data[['Age']])
data[['Age']] = imputer.transform(data[['Age']])
imputer = imputer.fit(data[['Salary']])
data[['Salary']] = imputer.transform(data[['Salary']])
print(data)
