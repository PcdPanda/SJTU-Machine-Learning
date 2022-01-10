import numpy as np
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt

# df = pd.DataFrame(pd.read_csv(r"D:\PANDA\Study\VG441\Homework\Problem Set 2\demand.csv"))
# d = df.d_t.T
# T, K, h = 52, 1100, 2.4
# M = 10e5
# # 导入数据
# WW = Model()
# q = WW.addVars(T, lb=np.zeros(T), vtype=GRB.CONTINUOUS, name="order_quantity")
# x = WW.addVars(T, lb=np.zeros(T), vtype=GRB.CONTINUOUS, name="inventory_level")
# y = WW.addVars(T, vtype=GRB.BINARY, name="if_order")

# WW.setObjective(quicksum(K*y[t]+h*x[t] for t in range(T)), GRB.MINIMIZE)

# c1 = WW.addConstrs(q[t] <= M*y[t] for t in range(T))
# c2 = WW.addConstrs(x[t] == x[t-1] + q[t] - d[t] for t in range(1,T))
# c3 = WW.addConstr(x[0] == q[0] - d[0])
# WW.optimize()
# # WW.printAttr('X')
# print(WW.getAttr('X',q).values())

# t=np.linspace(0,T-1,T)
# X=[234.9999999999999, 122.99999999999835, 0.0, 297.0, 150.0, 0.0, 174.0, 0.0, 197.0, 0.0, 206.99999999999997, 0.0, 240.00000000000006, 0.0, 241.0000000000011, 0.0, 267.0, 0.0, 289.9999999999999, 0.0, 301.0, 1.4203321069368455e-12, 324.9999999999985, 0.0, 343.0000000000001, 0.0, 340.0, 0.0, 323.0, 0.0, 309.0, 0.0, 293.0, 0.0, 272.9999999999999, 0.0, 248.99999999999994, 0.0, 234.99999999999994, 0.0, 204.0, 0.0, 182.99999999999994, 0.0, 163.99999999999994, 0.0, 299.0, 145.0, 0.0, 224.0, 112.0, 0.0]

# Q=[342.9999999999999, 0.0, 1.2505552149377763e-12, 426.0000000000004, 0.0, 0.0, 330.0, 0.0, 386.0, 0.0, 409.0, 0.0, 468.00000000000006, 0.0, 483.00000000000216, 0.0, 533.0, 0.0, 559.9999999999999, 0.0, 604.0, 1.4203321069368455e-12, 653.9999999999972, 1.553065288008162e-12, 688.0000000000001, 0.0, 707.0, 0.0, 660.0, 0.0, 634.0, 0.0, 598.0, 0.0, 545.9999999999999, 0.0, 500.99999999999994, 0.0, 477.99999999999994, 0.0, 430.0, 0.0, 386.99999999999994, 0.0, 338.99999999999994, 0.0, 461.0, 0.0, 0.0, 361.0, 0.0, 0.0]
# plt.plot(t,X,color='blue',label='Inventory Level')
# plt.plot(t,Q,color='red',label='Order Quantity')
# plt.plot(t,d,color='green',label='Demand')
# plt.legend(loc='upper left', fontsize=25)
# plt.show()


WW = Model()
T=1
# q = WW.addVars(T, lb=np.zeros(T), vtype=GRB.CONTINUOUS, name="order_quantity")
# x = WW.addVars(T, lb=np.zeros(T), vtype=GRB.CONTINUOUS, name="inventory_level")
# y = WW.addVars(T, vtype=GRB.BINARY, name="if_order")
y = WW.addVars(T, vtype=GRB.BINARY, name="if_order")
WW.setObjective(y, GRB.MINIMIZE)
c1 = WW.addConstrs(5*x2<=15)
c2 = WW.addConstrs(6*x1+2*x2<=24)
c3 = WW.addConstrs(x1+x2<=5)
c4 = WW.addConstrs(x1>=0)
c5 = WW.addConstrs(x2>=0)
c6 = WW.addConstrs(y=2*x1+x2)
WW.optimize()
# WW.printAttr('X')
print(WW.getAttr('X',q).values())