import numpy as np
import matplotlib.pyplot as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

def aucCV(features,labels):
    model = LogisticRegression()
    scores = cross_val_score(model, features, labels, cv=10,scoring='roc_auc')
    
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    model = LogisticRegression()
    model.fit(trainFeatures,trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]
    
    return testOutputs

if __name__ == "__main__":
    data = np.loadtxt('spam1.csv',delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = features[0::2,:]
    trainLabels = labels[0::2]
    testFeatures = features[1::2,:]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    pl.subplot(2,1,1)
    pl.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Target')
    pl.subplot(2,1,2)
    pl.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Output (predicted target)')