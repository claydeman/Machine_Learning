#decision Tree
import math
def calcShannonEnt(dataSet):
    
    L=len(dataSet)
    labels=[]
    for i in range(L):
        labels.append(dataSet[i][-1])
    labels=list(set(labels))    
    types=[0 for i in range(len(labels))]
    for i in range(L):
        types[labels.index(dataSet[i][-1])]+=1   
    prop=list(map(lambda x:(float(x)/L),types))     
    ent=sum(list(map(lambda x:x*(-math.log(x,2)),prop)))
    return ent
def createDataset():
    dataset=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flipper']
    return dataset,labels
def splitData(dataSet,label,dim):        
    data=dict.fromkeys(label)    
    types=[]
    for j in range(len(dataSet)):
        types.append(dataSet[j][dim])
        types=list(set(types))
    temp=[[]for i in range(len(types))]
    for i in range(len(dataSet)):
        ind=types.index(dataSet[i][dim])
        temp[ind].append(dataSet[i])
        data[label[0]]=temp
    return data
def chooseBestFeature(dataSet,label,ind):
    originalent=calcShannonEnt(dataSet)
    currentent=0
    distanceent=0    
    bestFeature=0
    splitdata=0    
    L=len(dataSet)
    for i in ind:        
        splitdata=splitData(dataSet,[label[i]],i)       
        for j in range(len(splitdata[label[i]])):
            currentent+=((len(splitdata[label[i]][j])/L)*calcShannonEnt(splitdata[label[i]][j]))
        if originalent-currentent>distanceent:
            originalent=currentent
            distanceent=originalent-currentent
            bestFeature=label[i]
    return bestFeature
def justifylabel(x):    
    temp=[]
    for i in range(len(x)):        
        temp.append(x[i][-1])
    temp=list(set(temp))        
    if(len(temp)==1):
        return False
    else:
        return True
def createTree(dataSet,label,ind):
    if justifylabel(dataSet)!=True:
        return dataSet
    feature=chooseBestFeature(dataSet,label,ind)
    result=splitData(dataSet,[feature],label.index(feature))        
    for i in range(len(result[feature])):
        if justifylabel(result[feature][i]):   
            ind.remove(label.index(feature))
            result[feature][i]=createTree(result[feature][i],label,ind)                
    return result   
def Dtclassify(inputTree,featLabels,testVec):
    while type(inputTree)==dict:
        firstStr=list(inputTree.keys())[0]
        featIndex=featLabels.index(firstStr)
        if type(inputTree[firstStr][testVec[featIndex]])==list:
            return (inputTree[firstStr][testVec[featIndex]])[0][-1]
        else:
            inputTree=inputTree[firstStr][testVec[featIndex]]
            
            
#testData=[1,1]  
'''   
data=splitData(a,b,1)
feature=chooseBestFeature(a,b)
'''
a,b=createDataset()
result=createTree(a,b,list(range(len(b))))
#testResult=Dtclassify(result,b,[1,0])
'''
data=splitData(testData,b)
result=dict((key, value) for key, value in data.items() if key==feature)
'''