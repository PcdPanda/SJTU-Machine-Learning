import numpy as np
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt
d = [220, 155, 105, 90, 170, 210, 290]
T=len(d)
K, h = 1000, 1.2
M = 10e5
# 导入数据
WW = Model()
q = WW.addVars(T, lb=np.zeros(T), vtype=GRB.CONTINUOUS, name="order_quantity")
x = WW.addVars(T, lb=np.zeros(T), vtype=GRB.CONTINUOUS, name="inventory_level")
y = WW.addVars(T, vtype=GRB.BINARY, name="if_order")

WW.setObjective(quicksum(K*y[t]+h*x[t] for t in range(T)), GRB.MINIMIZE)

c1 = WW.addConstrs(q[t] <= M*y[t] for t in range(T))
c2 = WW.addConstrs(x[t] == x[t-1] + q[t] - d[t] for t in range(1,T))
c3 = WW.addConstr(x[0] == q[0] - d[0])
WW.optimize()
WW.printAttr('X')
t=np.linspace(0,T-1,T)
X=[350,195,90,0,500,290,0]
Q=[570,0,0,0,670,0,0]
plt.plot(t,X,color='blue',label='Inventory Level')
plt.plot(t,Q,color='red',label='Order Quantity')
plt.plot(t,d,color='green',label='Demand')
plt.legend(loc='upper left', fontsize=25)
plt.show()