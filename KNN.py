#Machine learning in Action
#KNN

import numpy as np
import os
def img2vector(filename):
    resultVector=np.zeros((1,1024))   
    f=open(filename)
    for i in range(32):
        line=f.readline()
        if line:
            for j in range(32):
                resultVector[0][j+32*i]=int(line[j])                
    return resultVector

def readfile(listdirec):
    L=os.listdir(listdirec)
    dataSet=np.zeros((len(L),1024))
    labels=np.zeros((len(L),1))
    for i in range(len(L)):
        direc=listdirec+'/'+L[i]
        dataSet[i]=img2vector(direc)
        labels[i][0]=int(L[i][0])
    return dataSet,labels

def handwritingClassTest(data,dataset,labels,K):
    tempdataSet=dataset.copy()
    templabels=labels.copy()
    for i in range(len(tempdataSet)):
        tempdataSet[i,:]=tempdataSet[i,:]-data
    tempdataSet=np.sum(np.square(tempdataSet),axis=1)
    distances=tempdataSet**0.5
    label=np.argsort(distances)
    temp=[]
    for i in range(K):
        temp.append(int(templabels[label[i]]))
    M=0
    Mvalue=0
    S=list(set(temp))
    for i in range(len(S)):
        count=temp.count(S[i])
        if count>M:
            M=count
            Mvalue=S[i]
    return Mvalue
def handwritingTest():
    listdirec='./data/trainingDigits'
    a,b=readfile(listdirec)
    listD='./data/testDigits'
    L=os.listdir(listD)
    count=0
    number=0
    for i in range(len(L)):
        tempdir=listD+'/'+L[i]
        tempdata=img2vector(tempdir)
        templabel=int(L[i][0])
        labelValue=handwritingClassTest(tempdata,a,b,10)    
        print(templabel,labelValue,i)
        if labelValue==templabel:
            count+=1
        number+=1
    return count/number

#listdirec='C:\\Users\\70613\\machinelearninginaction\\Ch02\\digits\\trainingDigits'
#a,b=readfile(listdirec)
#print(int(b[1933]))
#print(handwritingClassTest(a[1933,:],a,b,1))
print(handwritingTest())