import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

data=pd.read_csv(r"D:\VS_CODE_PROJECTS-NARESH-IT\PRACTICE_ML\USA_Housing.csv")
x=data.iloc[::,0:4].values
y=data.iloc[:,5].values


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.2)

models={"LinearRegression":LinearRegression(), 'KNN': KNeighborsRegressor()}
results=[]

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })
       

results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)