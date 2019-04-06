import numpy as np
from utils_logistic import loaddata,feature_scaling,sigmoid,accuracy,confusionmatrix

"""tanh function"""
def tanh(x):
    return 2*sigmoid(2*x)-1

def tanh_gradient(x):
    t=tanh(x)
    return 1-t**2

"""Logistic binary loss function"""
def loss_logistic_binary(a_L,y_act):
    return -(y_act*np.log(a_L)+(1-y_act)*np.log((1-a_L)))

"""Initialize parameters W1,b1,W2,b2"""
def initialize(nx,n1,m):
    #---nx: number of features----------
    #---n1: number of nodes at first layer---
    #---m: total number of training examples----
    initW1=np.random.randn(n1,nx)*0.01
    initb1=np.zeros(shape=(n1,1))
    initW2=np.random.randn(1,n1)*0.01
    initb2=np.zeros(shape=(1,1))
    return (initW1,initb1,initW2,initb2)

"""Forward propogation"""
def forward(X,param,activate="tanh"):
    (W1,b1,W2,b2)=param
    Z1=np.dot(W1,X)+b1
    A1=tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    return (A1,A2)

"""Calculate cost"""
def calculatecost(a_L,y_act):
    m=y_act.shape[1]
    #---Cost of binary logistic classification at the output layer
    loss=loss_logistic_binary(a_L,y_act)
    cost=(1/m)*np.sum(loss)
    return cost

"""Back propogation"""
def back(X,y,A1,A2,W2):
    m=y.shape[1]
    #---dW2 and db2-------
    dZ2=A2-y
    dW2=(1/m)*(np.dot(dZ2,A1.T))
    db2=(1/m)*(np.sum(dZ2,axis=1,keepdims=True))
    #---dW2 and db1----------
    dA1=np.dot(W2.T,dZ2)
    dZ1=dA1*(1-A1**2)
    dW1=(1/m)*(np.dot(dZ1,X.T))
    db1=(1/m)*(np.sum(dZ1,axis=1,keepdims=True))
    return (dW1,db1,dW2,db2)

"""Update parameters"""
def update(params,gradient,alpha):
    (W1,b1,W2,b2)=params
    (dW1,db1,dW2,db2)=gradient
    W1=W1-alpha*dW1
    b1=b1-alpha*db1
    W2=W2-alpha*dW2
    b2=b2-alpha*db2
    return (W1,b1,W2,b2)

class nn_one_layer():
    def __init__(self):
        self.describe="This is class of one layer neural network model."
        self.cost_history=[]

    def train(self,X,y,n1=4,iter=100,alpha=0.01,activate="tanh"):
        #---n1: number of nodes at layer 1-------
        #---iter: number of iteration--------
        #---alpha: learning rate-------
        #---activate: activation function used at layer 1---------

        """Initialize parameters"""
        nx=X.shape[0]
        m=X.shape[1]
        (W1,b1,W2,b2)=initialize(nx,n1,m)

        """Iteration optimize"""
        for i in range(iter):
            (A1,A2)=forward(X,(W1,b1,W2,b2))
            cost=calculatecost(A2,y)
            self.cost_history.append(cost)
            (dW1,db1,dW2,db2)=back(X,y,A1,A2,W2)
            (W1,b1,W2,b2)=update((W1,b1,W2,b2),(dW1,db1,dW2,db2),alpha=alpha)

        return (W1,b1,W2,b2),self.cost_history

    def predict(self,X,param,threshold=0.5):
        # --params is optimized W---
        (A1,prob)=forward(X,param)
        predict = (prob >= threshold).astype(int)
        return predict