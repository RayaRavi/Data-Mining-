from sklearn.preprocessing import LabelEncoder#FOR CATEGORICAL DATA
from sklearn.preprocessing import StandardScaler#FOR SCALING FEATURES
from sklearn.model_selection import train_test_split#SPLITTING DATASET INTO TRAINING AND TESTING SETS
from sklearn.impute import SimpleImputer#FOR DEALING WITH MISSING DATA
from sklearn.preprocessing import OneHotEncoder#FOR PROCESS OF ELIMINATING OR ENCODING THE DUMMY VALUES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#FOR GRAPHICAL REPRESENTATION
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


print("--------using isnull()----------\n")
o=data.isnull()
print(o)


print("------------------FOR CATAGORICAL DATA-----------------\n")
categorical=pd.get_dummies(data,prefix=["AgeCatagory"],columns=["Age"])
print(categorical)


print("-------------------THE INFORMATION OF DATASET Data.csv------------------------\n")
info=data.info()
print(info)


print("--------------------USING LbelEncoder PRINTING THE INDEPENDENT VARIABLES-----------------\n")
label=LabelEncoder()
x[:,0]=label.fit_transform(x[:,0])
print(x)



print("-------------PROCESS OF ELIMINATING OR ENCODING DUMMY VALUES USING OneHotEncoder FOR INDEPENDENT VARIABLES-------------\n")
dummy=pd.get_dummies(data['Country'])
print(dummy)


print("-------------PROCESS OF ELIMINATING OR ENCODING DUMMY VALUES USING OneHotEncoder FOR DEPENDENT VARIABLES-------------\n")
dummy=pd.get_dummies(data['Purchased'])
print(dummy)


print("-------------PROCESS OF ELIMINATING OR ENCODING DUMMY VALUES USING OneHotEncoder WITHOUT USING pandas-------------\n")
onehot=OneHotEncoder()
onehot.fit_transform(data.Country.values.reshape(-1,1)).toarray()


print("-----------SPLITTING DATASET INTO TRAINING AND TESTING SETS---------------")
y=data.iloc[:,-1:].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print("----------FOR x_train-----------")
print(x_train)
print("----------FOR y_train-----------")
print(y_train)
print("----------FOR x_test------------")
print(x_test)
print("----------FOR y_test------------")
print(y_test)


print("----------SCALING THE FEATURES---------")
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print("--------------for  x_train------------")
print(x_train)
print("--------------for x_test--------------")
print(x_test)
