# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: M SRINATH
RegisterNumber: 212222230147
*/
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Salary_EX7.csv')
data.head()
data.info()
data.isnull().sum()
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split as t
x_train,x_test,y_train,y_test=t(x,y,test_size=0.2,random_state=2)
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics as m
mse=m.mean_squared_error(y_test,y_pred)
mse
r2=m.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## HEAD
![image](https://github.com/Saravana-kumar369/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117925254/79321eef-2dce-45de-b4c9-786bfa39f861)

## MSE
![image](https://github.com/Saravana-kumar369/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117925254/7d005251-881e-4742-a272-541165f13377)

## R2
![image](https://github.com/Saravana-kumar369/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117925254/e6520f0c-bdc2-4e6c-9739-022b6d140136)

## Predicted value
![image](https://github.com/Saravana-kumar369/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117925254/fb2ea208-e7c7-4197-b1ef-dfae14edff17)

## Decisuion Tree
![image](https://github.com/Saravana-kumar369/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/117925254/6444e3e5-bf42-4a2a-939f-f7ede2375869)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
