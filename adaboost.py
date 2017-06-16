#usage:python ababoost
import numpy as np
class adaboost:
    def __init__(self,size,data,label):
        self.dataset=np.mat(data)
        self.label=np.mat(label)
        self.size=size        
        self.m=self.dataset.shape[0]
        self.n=self.dataset.shape[1]
        self.D=np.mat(np.ones((self.dataset.shape[0],1)))/self.m
    def classify(self,dim,thresh,lg):        
        predLabel=np.ones((self.m,1))        
        if lg=='lt':                       
            predLabel[self.dataset[:,dim]<=thresh]=-1
        else:
            predLabel[self.dataset[:,dim]>thresh]=-1
        return predLabel
    def buildStump(self):   
        #print(self.D)
        minerror=float('inf')  
        minLabel=0
        bestStump={}
        for i in range(self.n):
            minVal=np.min(np.mat(self.dataset)[:,i])
            maxVal=np.max(np.mat(self.dataset)[:,i])
            stepsize=(maxVal-minVal)/self.size
            for j in range(-1,self.size+1):
                thresh=minVal+j*stepsize
                for k in ['lt','gt']:
                    errmat=np.ones((self.m,1))
                    templabel=self.classify(i,thresh,k)
                    errmat[templabel==self.label.T]=0
                    error=self.D.T*np.mat(errmat)
                   # print("the dim:%d the thresh:%f the equality : %s error: %f"\
                   #       %(i,thresh,k,error))
                    if error<minerror:
                        bestStump['dim']=i
                        bestStump['inequality']=k
                        bestStump['threshold']=thresh
                        minerror=error
                        minLabel=templabel                        
        #print(float(minerror),minthresh,minLabel)
        return bestStump,minerror,minLabel      
    def adaBoost(self,numIt=40):        
        itere=0
        error=float('inf')
        weakclassifier=[]
        aggclassEst=np.zeros((self.m,1))
        bestStump,minerror,minLabel=self.buildStump()
        while error!=0 and itere<=numIt:    
           # print(minerror)       
            alpha=float(float(1)/2*np.log(float((1-minerror))/minerror))  
           # print(alpha)    
            aggclassEst+=alpha*minLabel
           # print(aggclassEst)
            erroragg=np.sign(aggclassEst)
           # print(erroragg)
            error=np.sum(np.multiply(erroragg!=self.label.T,np.ones((self.m,1))))/self.m
           # print('finalerror:',error)
            bestStump['alpha']=alpha    
            weakclassifier.append(bestStump)
            mark=np.ones((self.m,1))*(-1)
            mark[minLabel!=self.label.T]=1     
            alpha=alpha*mark
            self.D=np.multiply(self.D,np.exp(alpha))/np.sum(self.D)
            bestStump,minerror,minLabel=self.buildStump() 
            #print(minLabel)
            itere+=1    
           # print(itere,error)
        return weakclassifier,aggclassEst,error
class adaTest:
    def classify(self,dataset,dim,thresh,lg):
        m,n=dataset.shape
        predLabel=np.ones((m,1))        
        if lg=='lt':                       
            predLabel[dataset[:,dim]<=thresh]=-1
        else:
            predLabel[dataset[:,dim]>thresh]=-1
        return predLabel
    def adaClassify(self,classtodata,classifier):
        classtodata=np.mat(classtodata)        
        result=0
        for i in range(len(classifier)):
            result+=classifier[i]['alpha']*self.classify(classtodata,classifier[i]['dim'],\
                                  classifier[i]['threshold'],classifier[i]['inequality'])
        print(result)
        return np.sign(result)
def loadDataSet(filedir):
    dataSet=[]
    labelSet=[]
    fr=open(filedir)
    numFeat=len(fr.readline().split('\t'))
    for line in fr.readlines():
        data=line.split('\t')
        temp=[]
        for i in range(numFeat-1):
            temp.append(float(data[i]))
        dataSet.append(temp)
        labelSet.append(float(data[-1]))
    return dataSet,labelSet     
    
          


test=adaTest()
traindata,trainlabel=loadDataSet('./data/horseColicTraining2.txt') 
testdata,testlabel=loadDataSet('./data/horseColicTest2.txt') 
ada=adaboost(10,traindata,trainlabel)  
weakclassifier,aggclassEst,error=ada.adaBoost()
print(weakclassifier)     
result=test.adaClassify(testdata,weakclassifier)
print(result)
