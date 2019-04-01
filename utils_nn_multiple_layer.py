import numpy as np
from utils_logistic import loaddata,feature_scaling,sigmoid,accuracy,confusionmatrix

def ReLU(x):
    return np.maximum(x,np.zeros(shape=x.shape))

def ReLU_grad(x):
    return (x>0).astype(int)

class nn_multiple_layer():
    def __init__(self,structure):
        self.structure=structure
        self.W_params=[]
        self.b_params=[]

    def initialize(self,X):
        """W1 and b1"""
        self.W_params.append(np.random.randn(self.structure[0],X.shape[0])*0.01)
        self.b_params.append(np.zeros(shape=(self.structure[0],1)))
        """W2,b2 to W and b of second to the last layer"""
        for i in range(len(self.structure)-1):
            self.W_params.append(np.random.rand(self.structure[i + 1], self.structure[i]) * 0.01)
            self.b_params.append(np.zeros(shape=(self.structure[i + 1], 1)))
        """W and b of last layer"""
        self.W_params.append(np.random.randn(1,self.structure[-1])*0.01)
        self.b_params.append(np.zeros(shape=(1,1)))

        return tuple(self.W_params),tuple(self.b_params)

    def forward(self,A_l_minus_one,W_l,b_l,activate="relu"):
        Z_l=np.dot(W_l,A_l_minus_one)+b_l
        A_l=ReLU(Z_l)
        return Z_l,A_l

    def back(self,dA_l,Z_l,W_l,A_l_minus_one,m,activation='relu'):
        m=A_l_minus_one.shape[1]
        dZ_l=dA_l*ReLU_gradient(Z_l)
        dW_l=(1/m)*np.dot(dZ_l,(A_l_minus_one.T))
        db_l=(1/m)*np.sum(dZ_l,axis=1,keepdims=True)
        dA_l_minus_one=np.dot(W_l.T,dZ_l)
        return dW_l,db_l,dA_l




# layer=[3,4,3]
# nn=nn_multiple_layer(layer)
# X=np.ones(shape=(3,20))
# W,b=nn.initialize(X)
