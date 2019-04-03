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
        self.describe="This is neural network model of multiple hidden layers."

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

        return self.W_params,self.b_params

    def forward(self,A_l_minus_one,W_l,b_l,activate="relu"):
        Z_l=np.dot(W_l,A_l_minus_one)+b_l
        A_l=ReLU(Z_l)
        return Z_l,A_l

    def back(self,dA_l,Z_l,W_l,A_l_minus_one,m,activation='relu'):
        dZ_l=dA_l*ReLU_grad(Z_l)
        dW_l=(1/m)*np.dot(dZ_l,(A_l_minus_one.T))
        db_l=(1/m)*np.sum(dZ_l,axis=1,keepdims=True)
        dA_l_minus_one=np.dot(W_l.T,dZ_l)
        return dW_l,db_l,dA_l_minus_one

    def update(self,W,b,dW_list,db_list,L,alpha=0.01):
        for i in range(0,L):
            W[i]=W[i]-alpha*dW_list[i]
            b[i]=b[i]-alpha*db_list[i]
        return W,b

    def train(self,X,y,iter=100,alpha=0.01):
        #---Number of training set-----
        m=y.shape[1]
        #---Initiate W and b---------
        W,b=self.initialize(X)
        #---Number of layers---------
        L=len(self.structure)+1
        #---Create empty A and Z list--------
        A=[0]*L
        Z=[0]*L
        #---Create empty dW, db list--------
        dW_list=[0]*L
        db_list=[0]*L
        #---Append X to A[0]-------
        A[0]=X
        #=====================================
        #===Below starts forward and backward iteration===============
        for iteration in range(0,iter):
            #=====================================
            #=====================================
            #--Forward propogation---------------
            #=====================================
            #----Layler one to L-1----------
            for i in range(0,L-1):
                z,a=self.forward(A[i],W[i],b[i])
                Z[i]=z
                A[i+1]=a
            #======================================
            #----Layer L output sigmoid------------
            Z_L=np.dot(W[L-1],A[L-1])+b[L-1]
            Z[L-1]=Z_L
            a_L=sigmoid(Z_L)
            #======================================
            #======================================
            #--Back propogation-------------------
            #======================================
            #----From loss to layer L--------------
            dZ_L=a_L-y
            dW_L=(1/m)*np.dot(dZ_L,A[L-1].T)
            dW_list[L-1]=dW_L
            db_L=(1/m)*np.sum(Z[L-1],axis=1,keepdims=True)
            db_list[L-1]=db_L
            da_L_minus_one=np.dot(W[L-1].T,Z[L-1])
            da_current=da_L_minus_one
            #----From layer L to layer 1------------
            for i in range(L-2,-1,-1):
                dW,db,da_current=self.back(da_current,Z[i],W[i],A[i],m)
                dW_list[i]=dW
                db_list[i]=db
            #=========================================
            #=========================================
            #--Update parameters---------------------
            W,b=self.update(W,b,dW_list,db_list,L)
            #=========================================
            #=========================================

            #--Return optimized W and b
            return W,b

    def predict(self,X,optW,optb,threshold=0.5):
        # ---Number of layers---------
        L = len(self.structure) + 1
        a_curr=X
        for i in range(0,L-1):
            z, a_curr = self.forward(a_curr, optW[i], optb[i])
        prob = sigmoid(np.dot(optW[L-1], a_curr) + optb[L-1])
        y_pred = (prob >= threshold).astype(int)
        return y_pred



# layer=[3,4,3]
# nn=nn_multiple_layer(layer)
# X=np.ones(shape=(3,20))
# W,b=nn.initialize(X)
