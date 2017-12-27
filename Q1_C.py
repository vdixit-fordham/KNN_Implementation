from __future__ import division
import time
import numpy as np
import pandas as pd
from pandas import DataFrame

#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# This method is to fo Z score normalization for any DataFrame.
def normalizeTrainDF(dataFrame):
    dfNormalized = dataFrame.copy()
    colList = list(dataFrame.columns)
    #print(cols)
    
    for col in range(len(colList)):
        colMean = dataFrame[colList[col]].mean()
        colStd = dataFrame[colList[col]].std()
        #print(col,'= ', colMean)
        #print(col,'= ', colStd)
        dfNormalized[colList[col]] = (dataFrame[colList[col]] - colMean)/colStd
    
    return dfNormalized

# This method is to fo Z score normalization for Test DataFrame using TrainingDF.
def normalizeTestDF(testDataFrame, trainDataFrame):
    print(np.shape(testDataFrame))
    print(np.shape(trainDataFrame))
    dfNormalized = testDataFrame.copy()
    colList = list(testDataFrame.columns)
    #print(cols)

    for col in range(len(colList)):
        colMean = trainDataFrame[colList[col]].mean()
        colStd = trainDataFrame[colList[col]].std()
        #print(col,'= ', colMean)
        #print(col,'= ', colStd)
        dfNormalized[colList[col]] = (testDataFrame[colList[col]] - colMean)/colStd

    return dfNormalized
                     

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


# creating a DF for row label
testRowID = testDF[testDF.columns[0]].copy()
#print(testRowID)

# dropping the ID columns from DF
testDF.drop(testDF.columns[0] , axis=1, inplace=True)

# Creating a new DF for labels.
trainLable = trainDF[['class']].copy()
testLable = testDF[['Label']].copy()

#print(testLable)

# dropping the Label columns from both DF
trainDF.drop('class' , axis=1, inplace=True)
testDF.drop('Label' , axis=1, inplace=True)

# Result DataFrame
resultDF = pd.DataFrame(index=testRowID[:50].values)
#print(resultDF)

#print(testDF["f1"].mean())
#print(testDF["f1"].std())
#print(np.shape(testDF))
zScroreTime = time.clock()
trainDFNormalized = normalizeTrainDF(trainDF)  
testDFNormalized = normalizeTestDF(testDF,trainDF)
#print("Time Taken in Z-Score Normalization is = ", (zScroreTime-startTime))
    
distanceDF = getAllDistanceDF(trainDFNormalized , testDFNormalized)
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
    for row in testDFNormalized.itertuples(index=False, name='Pandas'):
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
        # Breake the loop after 50 rows.
        if indexCounter >= 50:
            break
    
    tmpList = { kValuesList[kValue] : predictedLabel}
    predictedTestLabel = pd.DataFrame(tmpList, index=testRowID[:50].values)

    #print(predictedTestLabel)
    resultDF[str(kValuesList[kValue])] = predictedTestLabel
    #resultDF.join(predictedTestLabel)
    #print(resultDF)
    
#print(resultDF)

resultDF.replace([1,0],['spam','no-spam'], inplace=True)

print(resultDF)

#print('Total Time taken for all K is ', (time.clock() - startTime))   
    