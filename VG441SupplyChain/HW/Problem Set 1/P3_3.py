import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import xgboost as xgb

df=pd.DataFrame(pd.read_csv(r"D:\PANDA\Study\VG441\Homework\Problem Set 1\Cal_Housing.csv"))


class_mapping={'NEAR BAY':0, 'INLAND':1}
df['ocean_proximity']=df['ocean_proximity'].map(class_mapping) # 字符串转数字
df=df.dropna(axis=0,how='any',inplace=False) # 删除数据中所有含有nan的行

X=df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]
Y=df[['median_house_value']]
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.8)


# XGBoost in action...
params = {'n_estimators': 500, "objective":"reg:linear",'colsample_bytree': 0.5,'learning_rate': 0.05,
                'max_depth': 5, 'alpha': 1}
model = xgb.XGBRegressor(**params)
model.fit(X_train,Y_train)
Y_predicted = model.predict(X_test)



print("Mean squared error: %.2f"% mean_squared_error(Y_test, Y_predicted))
print('R2 sq: ',r2_score(Y_test, Y_predicted))
fig, ax = plt.subplots()
ax.scatter(Y_test, Y_predicted, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()