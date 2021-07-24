from gurobipy import *
WW = Model()
WW.Params.NonConvex = 2
s = WW.addVars(3,lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype='C', name="x") # m sets
WW.setObjective(-3*s[0]*s[0]+s[1]*s[1]+2*s[2]*s[2]+2*(s[0]+s[1]+s[2]), GRB.MINIMIZE)
WW.addQConstr(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]==1)
WW.optimize()
print('s=',WW.getAttr('X',s).values())
