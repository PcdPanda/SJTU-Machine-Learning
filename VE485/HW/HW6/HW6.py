import numpy as np
import matplotlib.pyplot as plt
# loss function
def loss(A, b, X, delta):
    y = np.dot(A, X) + delta - b # 50*1
    return np.dot(y.T, y)

# GD algorithm
def GD(A, b, X0, delta):
    e = 1e-2
    a = 0.2
    beta=0.25
    k = 0
    X = X0
    g = 2*np.dot(A.T,np.dot(A,X) + delta - b)
    n = []
    m = []
    while np.linalg.norm(g)>e: # When the gradient is not small enough
        g=2*np.dot(A.T,np.dot(A,X) + delta - b) # Calculate the negative gradient 1000*1
        t=1   
        while 1: # Backtracking
            Xt = X-t*g # 1000*1
            if loss(A, b, Xt, delta) < loss(A, b, X, delta) + a*t*np.dot(g.T, (Xt - X)):
                break
            t *= beta
        X = Xt
        n.append(k)
        m.append(np.linalg.norm(np.dot(A,X) + delta - b))
        k += 1
    plt.plot(n, m)
    return [X,k]

# generate the dataset
delta = 0 * (100**2)*np.random.randn(50, 1)
X = np.random.randn(1000, 1)
A = np.random.randn(50, 1000)
b = np.dot(A, X) + delta

delta = 0 * delta
# find the optimal solution through pseudo inverse directly
xhat = np.linalg.inv(np.dot(A, A.T))
xhat = np.dot(A.T, xhat)
xhat = np.dot(xhat, b-delta)
# print the rank of A
print('The rank of A is', np.linalg.matrix_rank(A))
# find the solution through GD
Result = GD(A, b, xhat, delta)
print('error for x0=xhat is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
Result = GD(A, b, 0.5*xhat, delta)
print('error for x0=0.5xhat is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
Result = GD(A, b, 0.75*X, delta)
print('error for x0=0.75x is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
Result = GD(A, b, 0.25*X, delta)
print('error for x0=0.25x is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
X0 = np.random.randn(1000, 1)
Result = GD(A, b, X0, delta)
print('error for x0=random matrix is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
Result = GD(A, b, X+X0, delta)
print('error for x0=x+random matrix is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
X0 = np.ones((1000, 1))
Result = GD(A, b, X0, delta)
print('error for x0=1 is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
Result = GD(A, b, 0*X0, delta)
print('error for x0=0 is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

# find the solution through GD
Result = GD(A, b, 0.5*X0, delta)
print('error for x0=0.5 is', np.linalg.norm(X-Result[0]), ' with ', Result[1], 'iterations')

plt.xlabel('iterations')
plt.ylabel('loss')
# plt.axis([0,10])
plt.show()