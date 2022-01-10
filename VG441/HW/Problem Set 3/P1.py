import numpy as np
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt

# 读取数据
data = pd.DataFrame(pd.read_csv('D:\PANDA\Study\VG441\Homework\Problem Set 3\Data.csv'))
data = data.iloc[:, 1:]
m = data.shape[0]
n = data.shape[1]
# 初始化数据
X = []
t = []
for i in range(m): X.append(list(data.loc[i])) 
WW = Model()
s = WW.addVars(m, vtype=GRB.BINARY, name="set_if_selected") # m sets
y = WW.addVars(n, name="element_if_covered") # n elements covered
WW.setObjective(quicksum(s), GRB.MINIMIZE)
for i in range(0, m):
    t.append(WW.addVars(n))
    WW.addConstrs(t[i][j] == s[i]*X[i][j] for j in range(n))
WW.addConstrs(y[j] == max_(t[i][j] for i in range(m)) for j in range(n))
WW.addConstrs(y[j] == 1 for j in range(n))
WW.optimize()
print('s=',WW.getAttr('X',s).values())