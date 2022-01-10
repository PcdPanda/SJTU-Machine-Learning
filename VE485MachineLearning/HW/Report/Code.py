import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def noise(r): # generate 500 noise
    return (np.random.rand(500) - 0.5) * np.random.rand() * r /5

def base():# generate a basis randomly for 3D Transformation
    n = np.random.rand(5)
    n = n - 0.5
    x = [n[0], n[1], n[2]]
    y = [n[3], n[4]]
    norm = np.linalg.norm(x)
    for i in range(0, len(x)):
        x[i] = x[i] / norm
    y.append((-x[0]*y[0] - x[1]*y[1]) / x[2])
    norm = np.linalg.norm(y)
    for i in range(0, len(y)):
        y[i] = y[i] / norm
    z = np.cross(x, y)  
    return([x, y, z])

def circle(O,r): # generate a circle
    angle = np.linspace(0, 2*3.1416, 500)
    x = O + np.cos(angle) * r + noise(r) 
    y = np.sin(angle) * r + noise(r) 
    z = np.zeros(len(x)) + noise(r) 
    return x, y, z

def dataset(r):  # generate dataset of two interlocked rings, 0.5<=r<=1, so the rings are coupled
    O1=1
    O2=0
    d = abs(O1 - O2)    
    angle = np.linspace(0, 2*3.1416, 250)
    x1, y1, z1 = circle(O1, r)
    x2, z2, y2 = circle(O2, r) # get two circles
    A = base()  # get a random 3D basis for transformation
    D1 = np.dot(A, np.array([x1, y1, z1])) 
    D2 = np.dot(A, np.array([x2, y2, z2])) # apply transformation
    return D1, D2

def draw_sca(ax, D1, D2, title='', lw =0.1): # drawing the scatter figure
    ax.scatter(D1[0], D1[1], D1[2], color='red', linewidths = lw)
    ax.scatter(D2[0], D2[1], D2[2], color='blue', linewidths = lw)
    ax.set_xlabel('x', color = 'red')
    ax.set_ylabel('y', color = 'blue')
    ax.set_zlabel('z', color = 'green')
    ax.set_title(title, x = 0.5, y = -0.1)

def SVM_train(D1, D2, model): # SVM Train 
    D1 = pd.DataFrame(np.c_[D1.T, np.zeros((len(D1[0]),1))])
    D2 = pd.DataFrame(np.c_[D2.T, np.ones((len(D2[0]),1))]) # Add label
    D = pd.concat([D1, D2])
    X = pd.DataFrame(preprocessing.minmax_scale(D.iloc[:, :-1], feature_range=(-0.5, 0.5)))  # normalization
    Y = D.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.9)
    model.fit(x_train, y_train)
    predict = model.predict(X)
    R = []
    B = []
    for i in range(predict.shape[0]): # label the result
        R.append([X.iloc[i,0], X.iloc[i,1], X.iloc[i,2]]) if predict[i] == 1 else B.append([X.iloc[i,0], X.iloc[i,1], X.iloc[i,2]])
    S = model.support_vectors_.T
    score = model.score(X, Y)
    return R, B, S, score

def SVM_Analysis(): #Performance Comparsion Analysis on different SVM model 
    r = np.linspace(1, 0.5, 25) #generte different x for different P
    y = [[],[]]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('SVM Performace Analysis score vs. p')
    for i in range(len(r)): 
        s=[[0],[0]]
        print(i)
        for j in range(10): # for each p, we test the two svm models for 10 times
            D1, D2 = dataset(r[i])
            R, B, S, score = SVM_train(D1, D2, svm.SVC(kernel='rbf')) 
            s[0] += score
            R, B, S, score = SVM_train(D1, D2, svm.SVC(kernel='rbf', gamma=25))
            s[1] += score
        y[0].append(s[0]/10)
        y[1].append(s[1]/10)
    r = 2 * x - 1 # calculate the p
    ax.set_ylim([0.9, 1])
    ax.set_xlim([1,0])
    ax.plot(r, y[0],label='gamma=default') 
    ax.plot(r,y[1],label='gamma=25')
    ax.legend( fontsize=15)
    plt.show()

def SVM(r): # generate the dataset, train the SVM model, and draw the classification result and support vector
    SVM_Analysis() # draw the score vs different p

    fig = plt.figure()  
    r=0.8
    D1, D2 = dataset(r)
    plt.title('p='+str(round(2*r-1,2))) # set r and p and generate data for specific SVM training

    R, B, S, score = SVM_train(D1, D2, svm.SVC(kernel='rbf', gamma=250)) # different gamma for KSVM
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    draw_sca(ax,np.array(R).T, np.array(B).T,'SVM score=' + str(score))
    ax.scatter(S[0], S[1], S[2], color='yellow', linewidths = 1)

    R, B, S, score = SVM_train(D1, D2, svm.SVC(kernel='rbf', gamma=25))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    draw_sca(ax,np.array(R).T, np.array(B).T,'SVM score=' + str(score))
    ax.scatter(S[0], S[1], S[2], color='yellow', linewidths = 1)
    plt.show() # draw two model classification result

def PCA_train(D1, D2, model): # train a PCA model
    D1 = pd.DataFrame(D1.T)
    D2 = pd.DataFrame(D2.T) # Add label
    D = pd.concat([D1, D2])
    X = D.iloc[:, :-1]
    X = pd.DataFrame(preprocessing.minmax_scale(X, feature_range=(-0.5, 0.5)))  # normalization
    X = pd.DataFrame(model.fit_transform(X))
    R=[]
    B=[]
    for i in range(X.shape[0]): # label the result
        R.append([X.iloc[i,0], X.iloc[i,1], X.iloc[i,2]]) if i < X.shape[0]/2 else B.append([X.iloc[i,0], X.iloc[i,1], X.iloc[i,2]])
    return R, B

def PCA(r):#kernel PCA Analysis based on different kernel and parameters
    fig = plt.figure()  
    D1, D2 = dataset(r)
    plt.title('p='+str(round(2*r-1,2))) # set r and p and generate data for specific SVM training

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    draw_sca(ax, D1, D2,'Original Data',0.1)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    R, B = PCA_train(D1, D2, KernelPCA(kernel='rbf',n_components=3,gamma=1/3))# default gamma
    draw_sca(ax, np.array(R).T, np.array(B).T,'KPCA result with gamma=default',0.1)
    
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    R, B = PCA_train(D1, D2, KernelPCA(kernel='rbf',n_components=3,gamma=25))# tuned gamma
    draw_sca(ax, np.array(R).T, np.array(B).T,'KPCA result with gamma=25',0.1)

    plt.show()

if __name__ == "__main__":
    SVM(0.6) # analyze different SVM model on different gamma
    PCA(0.55) # analyze different PCA model
    
    