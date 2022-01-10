from sympy import *
import numpy as np
import matplotlib.pyplot as plt
x = Symbol('x')
fx = log(exp(x)+exp(-x))
# fx = -log(x) + x
t = 1
x0 = 3
fdx = diff(fx,x)
fddx = diff(fdx,x)
e = 1e-5
k=1
while 1 :
 
    tx = fx.subs(x,x0)+fdx.subs(x,x0)*(x-x0)+fddx.subs(x,x0)*(x0-x0)**2/2
    x_nt = -fdx.evalf(subs = {'x':x0})/fddx.evalf(subs = {'x':x0})
    l=(x_nt**2)**0.5
    plt.figure(figsize=(20,20))
    plt.grid(1)
    # draw the points
    plt.scatter(x0,fx.evalf(subs = {'x':x0}),color='orange')
    title='The ' + str(k) + 'th iterations when x0 = ' + str(x0)
    x0 = x0+t*x_nt
    plt.scatter(x0,fx.evalf(subs = {'x':x0}),color='red')
    plt.scatter(x0,tx.evalf(subs = {'x':x0}),color='red')
    # draw the graph of functions
    n = np.linspace(float(x0)-5,float(x0)+5,100)
    Tx=[]
    Fx=[]
    Fdx=[]
    for i in range(len(n)) :
        Tx.append(tx.subs(x,n[i]))
        Fx.append(fx.subs(x,n[i]))
        Fdx.append(fdx.subs(x,n[i]))
    plt.plot(n, Fx, color='blue', label='f')
    plt.plot(n, Fdx, color='blue', label="f'", linestyle='--')
    plt.plot(n, Tx, label="Taylor Expansion")
    plt.legend(loc='upper left', fontsize=25) 
    print(n)
    plt.title(title)
    plt.show()

    if l**2/2 <= e:
        break
    k=k+1
