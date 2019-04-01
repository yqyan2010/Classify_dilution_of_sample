# import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
#
def loaddata():
    missing_values = ['n/a', 'na', 'Na', 'N/a', '-', '--']
    path=r"data/18E20a00.csv"
    df=pd.read_csv(path,na_values=missing_values)
    X=df.drop(["Not_filtered"], axis=1).values
    y=df["Not_filtered"].values
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.1, random_state=29)
    X_train=X_train.T
    X_test=X_test.T
    y_train=y_train.reshape(1,len(y_train))
    y_test=y_test.reshape(1,len(y_test))
    return X_train,X_test,y_train,y_test
#
def feature_scaling(a):
    #--substract mean and divide std----------
    #--scaling along each row
    a_mean=np.mean(a,axis=1,keepdims=True)
    a_std=np.std(a,axis=1,keepdims=True)
    a_scaled=(a-a_mean)/a_std
    return a_scaled

def sigmoid(x):
    return 1/(1+(np.exp(-x)))

def sigmoid_gradient(x):
    s=sigmoid(x)
    return s*(1-s)
#
# """L1 loss"""
# def L1_loss(y_pred,y_act):
#     return abs(y_pred-y_act)
#
# """L2 loss"""
# def L2_loss(y_pred,y_act):
#     return (y_pred-y_act)**2
#
# def initialize(size):
#     return np.random.random((1,size))
#
# def propogate(X,y,W):
#     m=X.shape[1]
#     Z=np.dot(W,X)
#     A=sigmoid(Z)
#     cost=np.sum(L2_loss(A,y))/m
#     dW=(np.dot(A-y,X.T))/m
#     return dW,cost
#
# def updateW(W,dW,alpha):
#     new_W=W-alpha*dW
#     return new_W
#
# def optimize(X,y,alpha=0.01,iter=100):
#     """Initialize W"""
#     n_feature=X.shape[0]
#     W=initialize(n_feature)
#
#     """Create a list to save cost"""
#     costlist=[]
#
#     """Iteration to optimize"""
#     for i in range(iter):
#         """Calculate dW and cost"""
#         dW,cost=propogate(X,y,W)
#         costlist.append(cost)
#         """Update W"""
#         W=updateW(W,dW,alpha)
#
#     return W,costlist
#
# def predict(X,param,threshold=0.5):
#     #--params is optimized W---
#     prob=sigmoid(np.dot(param,X))
#     predict=(prob>=threshold).astype(int)
#     return predict
#
"""Accuracy of model"""
def accuracy(y_pred,y_act):
    count=0
    #--Number of samples---
    m=y_act.shape[1]
    for i in range(m):
        if y_pred[0,i]==y_act[0,i]:
            count+=1
    return count/m

"""Confusion matrxi to evaluate the model"""
def confusionmatrix(y_pred,y_act):
    (tp,fp,tn,fn)=(0,0,0,0)
    #--Number of testing sample--
    m=y_act.shape[1]
    for i in range(m):
        if y_act[0,i]==0:#--actual N--
            if y_pred[0,i]==0:
                tn+=1
            else:
                fp+=1
        elif y_act[0,i]==1:#--actual P--
            if y_pred[0,i]==0:
                fn+=1
            else:
                tp+=1
    return (tp,fp,tn,fn)

# a=np.arange(9).reshape(3,3)
# print(a)
# b=np.arange(3).reshape(1,3)
# print(b)
#
# print(np.multiply(a,b))