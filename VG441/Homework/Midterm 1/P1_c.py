import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

df = pd.DataFrame(pd.read_csv(r"D:\PANDA\Study\VG441\Homework\Midterm 1\demands.csv"))
T=df[['time']]
D=df[['demands']]
n=50
X_train,X_test,Y_train,Y_test=train_test_split(T, D, test_size=0.8)

params = {'n_estimators': n, 'max_depth': 1, 'learning_rate': 1, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(X_train, Y_train)
model_score = model.score(X_train,Y_train)
Y_predicted = model.predict(X_test)
print("Mean squared error: %.2f"% mean_squared_error(Y_test, Y_predicted))
print('R2 sq: ',r2_score(Y_test, Y_predicted))

D_predicted = model.predict(T)
title='Gradient Boost Result with n_estimators='+str(n)
plt.title(title)
plt.scatter(T,D_predicted,label='Predicted Demands')
plt.scatter(T,D,label='Actual Demands')
plt.legend(loc='upper left', fontsize=25)
plt.show()