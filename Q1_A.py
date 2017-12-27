from __future__ import division
import time
import numpy as np
import pandas as pd
from pandas import DataFrame

#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def getAllDistanceDF(trainDF , testDF):
    indexCounter = 0;
    appenedData = []
    for row in testDF.itertuples(index=False, name='Pandas'):
        #print(indexCounter)
        #print(row)
        #print(type(row))
        distanceSeries = calculateEculidDist(trainDF, row)
        testRowIndexList = [indexCounter]*np.shape(distanceSeries.values)[0]
        #print(testRowIndexList)
        indexCounter += 1
    
        listToAddInFinalDF = { 'trainRowIndex' : np.array(distanceSeries.index) , 'distance': distanceSeries.values, 'testRowIndex' : testRowIndexList}
        dfEachTestRowDistance = pd.DataFrame(listToAddInFinalDF)
        appenedData.append(dfEachTestRowDistance)
    
    distanceDF = pd.concat(appenedData, axis=0)
    
    return distanceDF
    
def calculateEculidDist(trainDF , testRow):
    #np.sqrt(np.sum(np.square((trainArray - testArray))))
    tmp = (((trainDF.sub( testRow, axis=1))**2).sum(axis=1))**0.5
    tmp.sort_values(axis=0, ascending=True, inplace=True)
    #print(type(tmp))
    
    return tmp

startTime = time.clock()
#trainDF = pd.read_csv("spam_train.csv", nrows=10)
#testDF = pd.read_csv("spam_test.csv", nrows=10)
trainDF = pd.read_csv("spam_train.csv")
testDF = pd.read_csv("spam_test.csv")


# dropping the ID columns from DF
testDF.drop(testDF.columns[0] , axis=1, inplace=True)

# Creating a new DF for labels.
trainLable = trainDF[['class']].copy()
testLable = testDF[['Label']].copy()

#print(testLable)

# dropping the Label columns from both DF
trainDF.drop('class' , axis=1, inplace=True)
testDF.drop('Label' , axis=1, inplace=True)

distanceDF = getAllDistanceDF(trainDF , testDF)
distanceTime = time.clock()
#print('Time taken in Distance is = ', (distanceTime - startTime) )
#print(distanceDF)
#distanceDF.to_csv('distanceDF.csv')

kValuesList= [1,5,11,21,41,61,81,101,201,401]

# Running a outer loop for all K Values
accuracyList = []
for kValue in range(len(kValuesList)) :
    loopStartTime = time.clock()
    indexCounter = 0
    predictedLabel = []
    for row in testDF.itertuples(index=False, name='Pandas'):
        distanceDFForRow = distanceDF[ distanceDF['testRowIndex'] == indexCounter]
        #print(distanceDFForRow.loc[:4 ,'trainRowIndex'])
        nnIndex = distanceDFForRow.loc[:(kValuesList[kValue] - 1) ,'trainRowIndex']
    
        #print(trainLable.iloc[nnIndex])
        #print(type(trainLable.iloc[nnIndex]))
        tmp = trainLable.iloc[nnIndex]['class'].value_counts()
        #print(tmp)
        #print(type(tmp))
        #print(tmp.idxmax())
    
        predictedLabel.append(tmp.idxmax())
        indexCounter += 1
    
    tmpList = {'Label' : predictedLabel}
    predictedTestLabel = pd.DataFrame(tmpList)

    #print(predictedTestLabel)

    differenceLabel = testLable.sub(predictedTestLabel , axis=1)
    #print(differenceLabel)
    accurateClassCount = len(differenceLabel[ differenceLabel['Label'] ==0 ])
    #print(accurateClassCount)
    accuracyPercent = accurateClassCount/testLable['Label'].count()*100
    print('accuracy for k-' ,kValuesList[kValue] , 'is == ' , (accurateClassCount/testLable['Label'].count())*100, '%' )
    
    accuracyList.append(accuracyPercent)
    #predictedTestLabel.to_csv('predictedLable.csv')
    #print('Time taken for K-', kValuesList[kValue] ,' is ', (time.clock() - loopStartTime) , '\n\n\n\n')

tempDict = {'KValue' : kValuesList, 'Accuray %' : accuracyList}
accuracyDF = pd.DataFrame(tempDict)

#print(accuracyDF)

#print('Total Time taken for all K is ', (time.clock() - startTime))