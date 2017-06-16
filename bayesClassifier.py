#bayes classifier
import numpy as np
import math
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
        'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', \
         'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', \
         'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
         'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList,classVec            
def createVablist(dataset):
    data=[]
    for i in range(len(dataset)):
        data.extend(dataset[i])
    return list(set(data))
def setOfword2vec(vocabList,inputSet):
    vocabIndex=[0]*len(vocabList)
    for x in inputSet:
        vocabIndex[vocabList.index(x)]=1
    return vocabIndex

def trainNb0(trainMatrix,trainCategory):
    traindataSet=createVablist(trainMatrix)
    category=list(set(trainCategory))
    prop=[]
    for i in category:
        num=trainCategory.count(i)
        prop.append(num/len(trainCategory))
    p=[[1 for i in range(len(traindataSet))]for i in range(len(category))]
    pNum=[2 for i in range(len(category))]
    for i in range(len(p)):
        ind=[j for j,v in enumerate(trainCategory) if v==category[i]]        
        for x in ind:
            tempvocabIndex=setOfword2vec(traindataSet,trainMatrix[x])
            p[i]=list(map(lambda a,b:a+b,p[i],tempvocabIndex))
        for j in range(len(p[i])):
            p[i][j]=p[i][j]/((np.array(p[i])).sum()+pNum[i])
    return p,prop,category,traindataSet
def testNb0(testData,trainMatrix,trainCategory):
    p,prop,category,traindataSet=trainNb0(trainMatrix,trainCategory)
    testProp=[1 for i in range(len(category))]    
    for word in testData:
        for i in range(len(testProp)):
            testProp[i]+=(math.log(prop[i]*p[i][traindataSet.index(word)]))
    for i in range(len(testProp)):
        return category[testProp.index(max(testProp))]  

listOPosts,listClasses = loadDataSet()
myVocabList = createVablist(listOPosts)
trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(setOfword2vec(myVocabList, postinDoc))
a,b,c,d=trainNb0(listOPosts,listClasses)
testEntry = ['love', 'my', 'dalmation']
testNb0(testEntry,listOPosts,listClasses)