#simplified svm algorithm 
import numpy as np
import random
import matplotlib.pyplot as plt
class svmTest:
    def svmloadDataSet(self,filename):
        dataSet=[];labelMat=[]
        fr=open(filename)
        for line in fr.readlines():
            lineArr=line.split('\t')
            dataSet.append([float(lineArr[i]) for i in range(len(lineArr)-1)])
            labelMat.append(float(lineArr[-1]))
        return dataSet,labelMat
    def selectJrand(self,i,m):
        j=i
        while(j==i):
            j=int(random.uniform(0,m))
        return j
    def clipAlpha(self,aj,H,L):
        if aj>H:
            aj=H
        if L>aj:
            aj=L
        return aj
    def kernelTrans(self,X,A,kTup):
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
    def kernelRes(self,X,kTup):
        m=X.shape[0]
        kernel=np.mat(np.zeros((m,m)))
        for i in range(m):
            kernel[:,i]=self.kernelTrans(X,X[i,:],kTup)
        return kernel
    def smoSimplt(self,dataMatIn,classLabels,C,toler,maxIter,kTup):        
        dataMatrix=np.mat(dataMatIn);labelMat=(np.mat(classLabels)).T
        kernel=self.kernelRes(dataMatrix,kTup)                 
        b=0;m,n=dataMatrix.shape
        alphas=np.mat(np.zeros((m,1)))
        itere=0
        while itere<maxIter:
            alphaPairsChanged=0
            for i in range(m):
                fXi=float(np.multiply(alphas,labelMat).T*\
                          (dataMatrix*dataMatrix[i,:].T))+b
                Ei=fXi-float(labelMat[i])
                if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or\
                ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                    j=self.selectJrand(i,m)
                    fXj=float(np.multiply(alphas,labelMat).T*\
                              kernel[:,j])+b
                    Ej=fXj-float(labelMat[j])
                    alphaIold=alphas[i].copy()
                    alphaJold=alphas[j].copy()
                    if labelMat[i]!=labelMat[j]:
                        L=max(0,alphas[j]-alphas[i])
                        H=min(C,C+alphas[j]-alphas[i])
                    else:
                        L=max(0,alphas[j]+alphas[i]-C)
                        H=min(C,alphas[j]+alphas[i])
                    if L==H:print('L==H');continue
                    eta=2.0*kernel[i,j]-kernel[i,i]-kernel[i,j]
                    if eta>=0:print("eta>=0");continue
                    alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                    alphas[j]=self.clipAlpha(alphas[j],H,L)
                    if (abs(alphas[j]-alphaJold)<0.00001):
                        print("j not moving enough");continue
                    alphas[i]+=labelMat[j]*labelMat[i]*\
                         (alphaJold-alphas[j])
                    b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*\
                                    kernel[i,i]-\
                                    labelMat[j]*(alphas[j]-alphaJold)*\
                                    kernel[i,j]
                    b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                                    kernel[i,j]-\
                                    labelMat[j]*(alphas[j]-alphaJold)*\
                                    kernel[j,j]
                    if 0<alphas[i] and C>alphas[i]:b=b1
                    elif 0<alphas[j] and C>alphas[j]:b=b2
                    else:b=(b1+b2)/2.0
                    alphaPairsChanged+=1
                   # print("itere:%d i:%d,paris changed %d" %(itere,i,alphaPairsChanged))
            if alphaPairsChanged==0:itere+=1
            else :itere=0
            print("iteration number: %d" %itere)
        return b,alphas
    def showResult(self,dataMatIn,classLabels,b,alphas):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        for i in range(len(dataMatIn)):
            if classLabels[i]==-1:
                ax.scatter(dataMatIn[i][0],dataMatIn[i][1],c='r')
            else:
                ax.scatter(dataMatIn[i][0],dataMatIn[i][1],c='g')
        dataMatrix=np.mat(dataMatIn);labelMat=(np.mat(classLabels)).T    
        print(dataMatrix.shape)
        print(labelMat.shape)
        print(len(alphas))
        w=np.multiply(alphas,labelMat).T*dataMatrix
        print(w)
        x1=np.arange(0,10,0.1)
        x2=[(-b[0,0]-w[0,0]*x1[i])/w[0,1] for i in range(100)]
        ax.plot(x1,x2)
        plt.show()
test=svmTest()
data,labels=test.svmloadDataSet('./data/svm_testSet.txt')
b,alphas=test.smoSimplt(data,labels,0.6,0.001,40,('rbf',1.3))  
test.showResult(data,labels,b,alphas)   