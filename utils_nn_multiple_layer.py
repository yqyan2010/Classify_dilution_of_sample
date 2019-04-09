import numpy as np
from utils_logistic import loaddata,feature_scaling,sigmoid,accuracy,confusionmatrix
from utils_nn_one_layer import calculatecost

def ReLU(x):
    return np.maximum(x,np.zeros(shape=x.shape))

def ReLU_grad(x):
    return (x>0).astype(int)

#===Get list of W and b shape==============
def shapelist(W,b):#--Inputs are lists----
    W_shape=[]
    b_shape=[]
    #---W shape-------
    for i in range(0,len(W)):
        W_shape.append(W[i].shape)
    #---b shape-------
    for i in range(0,len(b)):
        b_shape.append(b[i].shape)
    return W_shape,b_shape

#===Convert to vectors====================
def Wb_to_vector(W,b):#--inputs are lists---------
    #---Initialize a theta with one 0--------
    theta=np.zeros((1,1))
    #---Vectorize W--------------------
    for i in range(0,len(W)):
        theta=np.concatenate((theta,W[i].reshape((W[i].size,1))),axis=0)
    #---Vectorize b--------------------
    for i in range(0,len(b)):
        theta=np.concatenate((theta,b[i].reshape((b[i].size,1))),axis=0)
    #---Remove first 0 in theta--------
    theta=theta[1:]
    return theta

#====Convert from vectors to W,b==========
def vector_to_Wb(theta,W_shape,b_shape):
    wb_size=0
    #---W total size---------
    for i in range(0,len(W_shape)):
        wb_size += W_shape[i][0]*W_shape[i][1]
    #---b total size----------
    for i in range(0,len(b_shape)):
        wb_size += b_shape[i][0]*b_shape[i][1]
    #---Checkpoint size needs to match---
    if wb_size != theta.size:
        print("The size of theta, and W,b do not match.")
    else:
        #---create empty list--------
        W=[]
        b=[]
        #---Convert to W-------------
        for i in range(0,len(W_shape)):
            size=W_shape[i][0]*W_shape[i][1]
            W.append(theta[0:size].reshape(W_shape[i]))
            #--IMPORTANT, slice theta after each iteration---
            theta=theta[size:]
        #---Convert to b-------------
        for i in range(0,len(b_shape)):
            size=b_shape[i][0]*b_shape[i][1]
            b.append(theta[0:size].reshape(b_shape[i]))
            theta=theta[size:]
        #---Checkpoint theta is empty at last
        if len(theta) != 0:
            print("Convert is complete but the size of theta, W and b do not match.")

    return W,b

def gradcheck(W,b,dW,db):#---inputs are lists-----------
    #---Convert W,b to theta, and dW, db to dtheta-------
    theta=Wb_to_vector(W,b)
    dtheta=Wb_to_vector(dW,db)
    #---Get W_shape, and b_shape-----------
    W_shape,b_shape=shapelist(W,b)
    #-----------------------------------
    #---Loops to calculate dtheta_aprox---------
    dtheta_aprox=np.zeros(dtheta.shape)
    theta_plus=theta
    theta_minus=theta
    for i in range(0,len(theta)):
        theta_plus[i]=theta[i]+epsilon
        theta_minus[i]=theta[i]-epsilon
        W_plus,b_plus=vector_to_Wb(theta_plus,W_shape,b_shape)
        W_minus,b_minus=vector_to_Wb(theta_minus,W_shape,b_shape)
        #=======================================================
        #===Forward to cost with W_plus,b_plus==================

    return diff

class nn_multiple_layer():
    def __init__(self,structure):
        self.structure=structure
        self.W_params=[]
        self.b_params=[]
        self.cost=[]
        self.dev_cost=[]
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

    def

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
            #--Calculate cost---------------------
            self.cost.append(calculatecost(a_L,y))
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
        return W,b,self.cost

    #===========================================================
    #=====Development and gradient checking=====================
    #===========================================================

    def train_with_dev(self,X,y,X_dev,y_dev,iter=100,alpha=0.01):
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
        #---Create empty diff_list----------
        diff_list=[]
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
            #--Calculate cost---------------------
            self.cost.append(calculatecost(a_L,y))
            #--Calculate dev cost-----------------
            a_curr=X_dev
            for i in range(0, L-1):
                z, a_curr = self.forward(a_curr, W[i], b[i])
            prob = sigmoid(np.dot(W[L-1], a_curr) + b[L-1])
            self.dev_cost.append(calculatecost(prob,y_dev))
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
            #===Gradient checking=====================
            diff=gradcheck(W,b,dW_list,db_list)
            diff_list.append(diff)
            #=========================================
            #=========================================
            #--Update parameters---------------------
            W,b=self.update(W,b,dW_list,db_list,L)
            #=========================================
            #=========================================

        #--Return optimized W and b
        return W,b,self.cost,self.dev_cost,diff_list

    def predict(self,X,optW,optb,threshold=0.5):
        # ---Number of layers---------
        L = len(self.structure) + 1
        a_curr=X
        for i in range(0,L-1):
            z, a_curr = self.forward(a_curr, optW[i], optb[i])
        prob = sigmoid(np.dot(optW[L-1], a_curr) + optb[L-1])
        y_pred = (prob >= threshold).astype(int)
        return y_pred

#=================================================================================
#===Test==========================================================================
#=================================================================================
# a=np.zeros((3,1))
# e=np.ones((4,1))print(theta.shape)
# print(a)
# print(b)
# print(theta)
# c=np.random.random((2,3))
# d=np.random.randn(3,4)
# W=[c,d]
# b=[a,e]
# W_shape,b_shape=shapelist(W,b)
# print(W_shape)
# print(b_shape)
