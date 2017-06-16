#logistic regression
#usage: python -i logisticReg.py

import numpy as np
import math
import matplotlib.pyplot as plt
def loadDataset(direc):
    DataSet=[]
    LabelSet=[]    
    data=np.genfromtxt(direc)    
    for i in range(len(data)):  
        temp=list(data[i][:-1])
        temp.insert(0,1)        
        DataSet.append(temp)
        LabelSet.append(data[i][-1])    
    return DataSet,LabelSet
def sigmoid(x,theta):
    h=x.dot(theta)    
    sig=np.zeros((h.shape[0],1))
    for i in range(len(x)):                
        sig[i,0]=1/(1+math.exp(-h[i,0]))    
    return sig
def LogRegression(direc,alpha):
    theta=[0,0,0]
    theta=(np.mat(theta)).T
    DataSet,LabelSet=loadDataset(direc)
    Data=DataSet[:]
    Data=np.mat(Data)    
    LabelSet=(np.mat(LabelSet)).T       
    T=0
    while T<100:        
        DataSet=sigmoid(Data,theta)
        error=LabelSet-DataSet
        temp=(Data.T).dot(error)*alpha
        theta=theta+temp  
       # print(theta)
        #showRegresult(theta)                  
        T+=1
    return theta    
    
def showRegresult(theta,direc):       
    theta=LogRegression(direc,0.01)
    DataSet,LabelSet=loadDataset(direc)
    DataSet=np.mat(DataSet)
    LabelSet=(np.mat(LabelSet)).T
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(len(DataSet)):
        if LabelSet[i,0]==0:
            xcord1.append(DataSet[i,1])
            ycord1.append(DataSet[i,2])
        else:
            xcord2.append(DataSet[i,1])
            ycord2.append(DataSet[i,2])            
    plt.plot(xcord1,ycord1,'ro')
    plt.plot(xcord2,ycord2,'b*')    
    i=np.linspace(-5,5,100)
    theta=np.mat(theta)
    plt.plot(i,(-0.5-theta[0,0]-theta[1,0]*i)/theta[2,0],'-',label='Logistic Regression')
    plt.legend(loc='upper left', shadow=True, fontsize='medium')
    plt.show()

    
direc='./data/testSet.txt'    
showRegresult(0.01,direc)