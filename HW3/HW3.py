import numpy as np, pandas as pd, csv, re

dataFrame = pd.read_csv("Homework 3 HistoricalQuotes.csv")
prices = dataFrame['Close/Last'].astype(str).str.replace(r'[$, ]','', regex=True).astype(float).to_numpy()[::-1]

def lmsOnlineMse(series, order=2, learningRate=0.05, useBias=False, testRatio=0.2):
    normalizedSeries = (series - series.mean())/(series.std()+1e-9)
    totalLength = len(normalizedSeries); splitIndex = int((1-testRatio)*totalLength)
    weights = np.zeros(order); bias = 0.0; errors = []
    for index in range(order, totalLength):
        features = normalizedSeries[index-order:index][::-1]
        trueValue = normalizedSeries[index]
        prediction = float(np.dot(weights, features) + (bias if useBias else 0.0))
        if index >= splitIndex: errors.append((prediction - trueValue)**2)
        error = trueValue - prediction
        weights = weights + learningRate*error*features
        if useBias: bias = bias + learningRate*error
    return float(np.mean(errors)), weights, (bias if useBias else 0.0)

def rescaleMse(normalizedMse, series):
    standardDeviation = series.std()
    return (standardDeviation**2) * normalizedMse

results = {}
for order in (2,3):
    for useBias in (False, True):
        normalizedMse, weights, bias = lmsOnlineMse(prices, order=order, learningRate=0.05, useBias=useBias, testRatio=0.2)
        results[(order,useBias)] = (rescaleMse(normalizedMse, prices), weights, bias)

print("LMS MSEs (lower is better):")
for (order,useBias),(mse,weights,bias) in results.items():
    print(f"  order{order}_{'bias' if useBias else 'nobias'} : MSE = {mse:.6g}")

bestLmsKey = min(results, key=lambda k: results[k][0])
print("\nBest LMS (by MSE):", f"order{bestLmsKey[0]}_{'bias' if bestLmsKey[1] else 'nobias'}",
      "=> MSE =", results[bestLmsKey][0])

def sigmoid(value): return 1/(1+np.exp(-value))

def buildFeatures(series, order=3):
    normalizedSeries = (series - series.mean())/(series.std()+1e-9)
    features=[]; targets=[]
    for index in range(order, len(series)):
        features.append(normalizedSeries[index-order:index][::-1])
        targets.append(1 if series[index] > series[index-1] else 0)
    return np.array(features,float), np.array(targets,int)

def trainLogisticRegression(features, targets, learningRate=0.2, iterations=4000, regularization=1e-3):
    numSamples, numFeatures = features.shape
    weights = np.zeros(numFeatures); bias = 0.0
    for _ in range(iterations):
        linearCombination = features@weights + bias
        probabilities = sigmoid(linearCombination)
        gradient = probabilities - targets
        weights -= learningRate * ((features.T @ gradient)/numSamples + regularization*weights)
        bias -= learningRate * np.mean(gradient)
    return weights, bias

def evaluateLogisticNumericProxy(series, order=3):
    features, targets = buildFeatures(series, order=order)
    splitIndex = int(0.8*len(features))
    trainFeatures, testFeatures = features[:splitIndex], features[splitIndex:]
    trainTargets, testTargets = targets[:splitIndex], targets[splitIndex:]
    weights, bias = trainLogisticRegression(trainFeatures, trainTargets)
    testProbabilities = sigmoid(testFeatures@weights + bias)
    accuracy = (testProbabilities>=0.5).astype(int).mean() == testTargets
    accuracy = ((testProbabilities>=0.5).astype(int) == testTargets).mean()
    returns = np.diff(series)
    avgAbsoluteReturn = np.mean(np.abs(returns))
    startIndex = order + splitIndex
    lastPrices = series[startIndex-1 : len(series)-1]
    expectedSign = (testProbabilities - 0.5)*2.0
    predictedPrices = lastPrices + expectedSign*avgAbsoluteReturn
    truePrices = series[startIndex:]
    mse = np.mean((predictedPrices - truePrices)**2)
    return accuracy, mse

accuracy, mseProxy = evaluateLogisticNumericProxy(prices, order=3)
print("\nLogistic regression (direction):")
print(f"  Accuracy (up/down) = {accuracy:.4f}")
print(f"  Numeric MSE proxy  = {mseProxy:.6g}")

allMse = {f"order{order}_{'bias' if useBias else 'nobias'}":mse for (order,useBias),(mse,_,_) in results.items()}
allMse['logistic_numeric_proxy'] = mseProxy
bestKey = min(allMse, key=allMse.get)
print("\nBest (by numeric MSE among all):", bestKey, "=> MSE =", allMse[bestKey])
print("Script execution completed successfully!")