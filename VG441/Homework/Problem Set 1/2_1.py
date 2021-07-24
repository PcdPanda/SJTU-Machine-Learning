import numpy as np
import pandas as pd
A=[40,1,1,1,10000]
B=[20,0,0,0,500]
C=[50,1,0,1,8000]
D=[30,1,0,0,5000]
Salary=[A[4],B[4],C[4],D[4]]
average=np.mean(Salary)
F=average
P0=Salary-average
SS=(sum(P0))**2/(len(Salary)+1)



Gain=(P0[0]+P0[2])**2/(2+1)+(P0[1]+P0[3])**2/(2+1)
print("Gain=",Gain)

SS=(P0[0])**2/(1+1)+(P0[2])**2/(1+1)-Gain
print("SS=",SS)
# SS=(sum(Salary)-len(Salary)*average)**2/(4+1+1)
# print("average=",average)
# print("SS=",SS)

# for i in range(len(Salary)):
#     deviance+=(Salary[i]-average)**2
# F=average
# print("FO=",F)
# for i in range(3):
#     P=D[4]-F
#     F+=0.1*P
#     print("F=",F," P=",P)
