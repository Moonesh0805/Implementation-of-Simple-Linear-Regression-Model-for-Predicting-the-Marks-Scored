# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph.
5.Predict the regression for the marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MOONESH P
RegisterNumber:212223230126


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
*/
```
## Output:
## df.head():
![image](https://github.com/user-attachments/assets/85c63e60-5c75-44ad-a3b9-bf3e45a55ca4)

## df.tail():
![image](https://github.com/user-attachments/assets/4e557252-a9ea-4dcf-a6af-6a9b7d82707b)

## Values of x:
![image](https://github.com/user-attachments/assets/2fb45f59-5b66-4298-81f3-8f546c285c14)

## Values of y:
![image](https://github.com/user-attachments/assets/21e4df14-9075-4358-963e-3120bb64ba55)

## Values of y prediction:
![image](https://github.com/user-attachments/assets/4b7f559b-a4d6-468b-826c-d73401e883d7)

## Values of y test:
![image](https://github.com/user-attachments/assets/1e77c027-f6a4-467c-9e91-79c82091f02e)

## Training set graph:
![image](https://github.com/user-attachments/assets/3c13338c-fec1-4bee-a4cc-477d3d88c9d8)

## Test set graph:
![image](https://github.com/user-attachments/assets/6e1b6acb-33a3-4b0d-b3b6-c1036f2b8467)

## Value of MSE,MAE & RMSE:
![image](https://github.com/user-attachments/assets/f92abeab-acb1-44d1-a7e1-8b563cfc1dce)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
