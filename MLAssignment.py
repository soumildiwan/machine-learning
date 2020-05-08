import numpy as np
import pandas as pd
from random import random
from random import seed
from math import exp
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



#DATA POINTS
sampleNum = 200

#FEATURES
featureNum = 4

#REDUNDENT
redundantNum = 1

#CLASSES
classesNum = 2

#READING Dataset
X, y = make_classification(n_samples=sampleNum, n_features=featureNum, n_redundant=redundantNum, n_classes=classesNum)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df['label'] = y
df.to_csv("dataset1.csv")
df=pd.read_csv('dataset1.csv',index_col=0)


#FUNCTION FOR TRANSFER
def transferFun(activater):
    return 1.0 / (1.0 + exp(-activater))


#FUNCTION FOR TRANSFERING DEVRIVATIVE
def transferDerivative(input):
    return input * (1.0 - input)


#PREDICTION FUNCTION
def predicting(net, row):
    otpt = forwardPropagate(net, row)
    return otpt.index(max(otpt))

#FUNCTION FOR ACTIVATION
def activateFun(wt, inp):
    activator=wt[-1]
    for i in range(len(wt)-1):
        activator+=wt[i]*inp[i]
    return activator




#NETWORK INITIALIZING
def networkInitializing(inputNum, hiddenNum, outputNum):
    net=list()
    hiddenLayer = [{'weights':[random() for i in range(inputNum + 1)]} for i in range(hiddenNum)]
    net.append(hiddenLayer)
    outputLayer = [{'weights':[random() for i in range(hiddenNum + 1)]} for i in range(outputNum)]
    net.append(outputLayer)
    return net



#FUNCTION FOR BACKWARD PROPOGATION
def backwardPropagate(net, expted):
    for i in reversed(range(len(net))):
        lyr = net[i]
        err = list()
        if i != len(net)-1:
            for j in range(len(lyr)):
                error = 0.0
                for neur in net[i + 1]:
                    error += (neur['weights'][j] * neur['delta'])
                err.append(error)
        else:
            for j in range(len(lyr)):
                neur = lyr[j]
                err.append(expted[j] - neur['output'])
        for j in range(len(lyr)):
            neur = lyr[j]
            neur['delta'] = err[j] * transferDerivative(neur['output'])



#FORWARD PORPOGATION
def forwardPropagate(net,data):
    rawInp=data
    for i in net:
        newRaw=[]
        for j in i:
            activater=activateFun(j['weights'], rawInp)
            j['output']=transferFun(activater)
            newRaw.append(j['output'])
        rawInp=newRaw
    return rawInp




#FUNCTION FOR TRAINING THE NETWORK
def trainNetworkFun(net, training, l_rate, epochNum, outputNum):
    for epoch in range(epochNum):
        sum_err = 0
        for row in training:
            otpt = forwardPropagate(net, row)
            expted = [0 for i in range(outputNum)]
            expted[int(row[-1])] = 1
            sum_err += sum([(expted[i]-otpt[i])**2 for i in range(len(expted))])
            backwardPropagate(net, expted)
            updatingWeights(net, row, l_rate)
        print('Loop=%d, learn_rate=%.3f, Error=%.3f' % (epoch, l_rate, sum_err))



#FUNCTIONING FOR UPDATING WEIGHTS
def updatingWeights(net, row, l_rate):
    for i in range(len(net)):
        inp=row[:-1]
        if i!=0:
            inp=[neur['output'] for neur in net[i-1]]
        for neur in net[i]:
            for j in range(len(inp)):
                neur['weights'][j]+=l_rate*neur['delta']*inp[j]
            neur['weights'][-1]+=l_rate*neur['delta']





#ARRAY IN DATASET
dataset=np.array(df[:])
dataset


#SETTING INPIT AND OUTPUT
inputNum = len(dataset[0]) - 1
outputNum = len(set([row[-1] for row in dataset]))
print(inputNum,outputNum)


#SPLIT DATASET
trainDatasetVar=dataset[:150]
testDatasetVar=dataset[150:]



#DATASET INTO NET
net=networkInitializing(inputNum,1,outputNum)
trainNetworkFun(net, trainDatasetVar, 0.5, 100, outputNum)



#WEIGHTS OF NETWORK
for lyr in net:
    print(lyr)



#TESTING DATASET
testSet=[]
pred=[]
for row in testDatasetVar:
    prediction = predicting(net, row)
    testSet.append(row[-1])
    pred.append(prediction)
print()
print("TEST DATASET")
print("Confusion Matrix is: ",confusion_matrix(testSet,pred))
print("Accuracy is: ",accuracy_score(testSet,pred))
print("Precision is: ",precision_score(testSet, pred))
print("recall is: ",recall_score(testSet, pred))



#TRAINING DATASET
trainSet=[]
pred=[]
for row in trainDatasetVar:
    prediction = predicting(net, row)
    trainSet.append(int(row[-1]))
    pred.append(prediction)

print()
print("TRAIN DATASET")
print("Confusion Matrix is: ",confusion_matrix(trainSet,pred))
print("Accuracy is: ",accuracy_score(trainSet,pred))
print("Precision is: ",precision_score(trainSet, pred))
print("recall is: ",recall_score(trainSet, pred))
