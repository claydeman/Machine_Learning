#full svm algorithm
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
def kernelTrans(X,A,kTup):
        m,n=X.shape
        K=np.mat(np.zeros((m,1)))
        if kTup[0]=='lin':K=X*A.T
        elif kTup[0]=='rbf':
            for j in range(m):
                deltaRow=X[j,:]-A
                K[j]=deltaRow*deltaRow.T
            K=np.exp(K/(-1*kTup[1]**2))
        else:raise NameError('The kernel is not recognized')              
        return K
class optConst:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.data=dataMatIn
        self.label=classLabels
        self.C=C
        self.tol=toler
        self.b=0
        self.m=dataMatIn.shape[0]
        self.caches=np.mat(np.zeros((self.m,2)))
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.kernel=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.kernel[:,i]=kernelTrans(self.data,self.data[i,:],kTup)
class fullsvm:
    def svmloadDataSet(self,filename):
        dataSet=[];labelMat=[]
        fr=open(filename)
        for line in fr.readlines():
            lineArr=line.split('\t')
            dataSet.append([float(lineArr[i]) for i in range(len(lineArr)-1)])
            labelMat.append(float(lineArr[-1]))
        return dataSet,labelMat
    def calcEk(self,op,k):
        fXk=float(np.multiply(op.alphas,op.label).T*\
                          op.kernel[:,k])+op.b
        Ek=fXk-op.label[k]
        return Ek
    def selectJrand(self,i,m):
        j=i
        while(j==i):
            j=int(random.uniform(0,m))
        return j
    def selectJ(self,op,i,Ei):
        op.caches[i]=[i,Ei]
        vaildEcacheList=np.nonzero(op.caches[:,0])[0]
        Ej=0;maxK=-1;maxDeltaE=0
        if len(vaildEcacheList)>1:
            for x in vaildEcacheList:
                if x==i:continue
                Ek=self.calcEk(op,x)
                if abs(Ei-Ek)>maxDeltaE:                    
                    maxK=x;maxDeltaE=abs(Ei-Ek);Ej=Ek
                return maxK,Ej
        else:
            j=self.selectJrand(i,op.m)
            Ej=self.calcEk(op,j)
        return j,Ej    
    
    def updateEk(self,op,i):
        Ek=self.calcEk(op,i)
        op.caches[i]=[i,Ek]
    def clipAlpha(self,aj,H,L):
        if aj>H:
            aj=H
        if L>aj:
            aj=L
        return aj    
        
    def inner(self,i,op):
        Ei=self.calcEk(op,i)
        if ((op.label[i]*Ei < -op.tol) and (op.alphas[i] < op.C)) or\
                     ((op.label[i]*Ei > op.tol) and (op.alphas[i] > 0)):
            j,Ej = self.selectJ(op,i, Ei)
            alphaIold = op.alphas[i].copy(); alphaJold = op.alphas[j].copy();
            if (op.label[i] != op.label[j]):
                L = max(0, op.alphas[j] - op.alphas[i])
                H = min(op.C, op.C + op.alphas[j] - op.alphas[i])
            else:
                L = max(0, op.alphas[j] + op.alphas[i] - op.C)
                H = min(op.C, op.alphas[j] + op.alphas[i])
            if L==H: print("L==H"); return 0
            eta = 2.0 * op.kernel[i,j] - op.kernel[i,i] -op.kernel[j,j]                   
            if eta >= 0: print("eta>=0"); return 0
            op.alphas[j] -= op.label[j]*(Ei - Ej)/eta
            op.alphas[j] = self.clipAlpha(op.alphas[j],H,L)
            self.updateEk(op, j)
            if abs(op.alphas[j]-alphaJold)<0.00001:
                print("J not moving enough");return 0
            op.alphas[i]+=op.label[j]*op.label[i]*(alphaJold-op.alphas[j])
            self.updateEk(op,i)            
            b1 = op.b - Ei- op.label[i]*(op.alphas[i]-alphaIold)*\
                 op.kernel[i,i] - op.label[j]*\
                 (op.alphas[j]-alphaJold)*op.kernel[i,j]
            b2 = op.b - Ej- op.label[i]*(op.alphas[i]-alphaIold)*\
                 op.kernel[i,j] - op.label[j]*\
                 (op.alphas[j]-alphaJold)*op.kernel[j,j]
            if (0 < op.alphas[i]) and (op.C > op.alphas[i]): op.b = b1
            elif (0 < op.alphas[j]) and (op.C > op.alphas[j]): op.b = b2
            else: op.b = (b1 + b2)/2.0
            return 1
        else: return 0
    def smoP(self,dataMatIn,labels,C,toler,maxIter,kTup):
        op=optConst(np.mat(dataMatIn),np.mat(labels).T,C,toler,kTup)
        itere=0
        entireSet=True;alphaPairsChanged=0       
        while itere<maxIter and (alphaPairsChanged>0 or entireSet):
            alphaPairsChanged=0
            if entireSet:
                for i in range(op.m):
                    alphaPairsChanged+=self.inner(i,op)
               # print("fullSet, iter:%d i:%d,pairs changed %d" %(itere,i,alphaPairsChanged))
                itere+=1
            else:
                nonBoundIs=np.nonzero((op.alphas.A>0)*(op.alphas.A<C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged+=self.inner(i,op)
                #print("non-bound,iter: %d i:%d,pairs changed %d" %(itere,i,alphaPairsChanged))
                itere+=1
            if entireSet:entireSet=False
            elif alphaPairsChanged==0:entireSet=True
            print("iteration number:%d" %itere)
        self.testRbf(op)
        return op.b,op.alphas
    def showResult(self,dataMatIn,classLabels,b,alphas):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        for i in range(len(dataMatIn)):
            if classLabels[i]==-1:
                ax.scatter(dataMatIn[i][0],dataMatIn[i][1],c='r')
            else:
                ax.scatter(dataMatIn[i][0],dataMatIn[i][1],c='g')
        dataMatrix=np.mat(dataMatIn);labelMat=(np.mat(classLabels)).T    
        w=np.multiply(alphas,labelMat).T*dataMatrix
        print(w)
        x1=np.arange(0,10,0.1)
        x2=[(-b[0,0]-w[0,0]*x1[i])/w[0,1] for i in range(100)]
        ax.plot(x1,x2)
        plt.show()   
    def testRbf(self,op,k1=1.3):        
        nozeroaind=np.nonzero(op.alphas.A>0)[0]
        nozeroamat=op.data[nozeroaind]
        nozerobmat=op.label[nozeroaind]       
        errorCount=0
        for i in range(op.m):
            kernelEval=kernelTrans(nozeroamat,op.data[i,:],('rbf',k1))
            predictVal=kernelEval.T*np.multiply(nozerobmat,op.alphas[nozeroaind])+op.b
            if pd.np.sign(predictVal)!=pd.np.sign(op.label[i]):
                errorCount+=1
        print(float(errorCount)/op.m)


fulltest=fullsvm()
data,labels=fulltest.svmloadDataSet('./data/svm_testSet.txt') 
b,alphas=fulltest.smoP(data,labels,0.6, 0.001, 60,['lin'])           
fulltest.showResult(data,labels,b,alphas)