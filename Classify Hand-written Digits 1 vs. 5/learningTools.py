import numpy as np
import yaml
import digitsExtractor

# Load Config info
with open("settings.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

# Vairable Declaration
max_epoches = data_loaded['max_epoches']
learning_rate = data_loaded['learning_rate']
testDict, testData, testLabel = digitsExtractor.extractFile("ZipDigits.test")
trainDict, trainData, trainLabel = digitsExtractor.extractFile("ZipDigits.train")
weight = np.zeros(3)
x = np.asarray(trainData)
y = np.array(trainLabel)
x_test = np.asarray(testData)

# Linear Regression with min square eq
def linearReg(x, weight, xTest):
    # Linear Regression with least square method
    pseudoInverseX = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x))
    wLin = np.dot(pseudoInverseX, y)
    hypothesis = np.dot(x, wLin)
    # Convert float result in hypothesis into 1 or 0
    hypothesis = np.asarray([int(element + 0.5) for element in hypothesis])
    eTrain = ((hypothesis - trainLabel) ** 2).mean(axis=None)

    # Validation
    hypoTest = np.dot(xTest, wLin)
    hypoTest = np.asarray([int(element + 0.5) for element in hypoTest])
    eTest = ((hypoTest - testLabel) ** 2).mean(axis=None)
    return eTrain, eTest

def perceptron():
    # PLA(Perceptron) with pocket algorithm as improvement
    weights = (1, 1)
    minEtrain = 100
    minWeight = weights
    for iter in range(max_epoches):
        for i in range(len(trainData)):
            # If the x1*w1+x2*w2>0, the number should be five(label = 0); If the x1*w1+x2*w2<0, label =1
            # else, re-balance the weights
            calculatedLabel = trainData[i][1] * weights[0] + trainData[i][2] * weights[1]
            if ((calculatedLabel > 0 and trainLabel[i] == 0) or (calculatedLabel < 0 and trainLabel[i] == 1)):
                continue
            else:
                aLabel = calculatedLabel
                aArray = np.asarray(trainData[i][1:])
                a = learning_rate * calculatedLabel * np.asarray(trainData[i][1:]) / (
                            trainData[i][1] ** 2 + trainData[i][2] ** 2)
                weights -= a
                # hypothesis = x1*w1+x2*w2
                hypothesis = np.dot(np.asarray(trainData)[:, 1:], np.asarray(weights))
                hypothesis = np.asarray([(0 if element > 0 else 1) for element in hypothesis])
                eTrain = ((hypothesis - trainLabel) ** 2).mean(axis=None)
                if eTrain < minEtrain:
                    minEtrain = eTrain
                    minWeight = weights
    print("The training error is %.5f" % minEtrain)
    # Validation
    hypoTest = np.dot(np.asarray(testData)[:, 1:], np.asarray(weights))
    hypoTest = np.asarray([(0 if element > 0 else 1) for element in hypoTest])
    eTest = ((hypoTest - testLabel) ** 2).mean(axis=None)
    print("The Test set error is %.5f" % eTest)


def logisticReg(x, weight, xTest):
    # Logistic Regression with Graident Descent
    tolerance = 0.02
    epoch = 0
    eTrain = 1
    while (eTrain > tolerance and epoch < max_epoches):
        z = np.dot(x, weight)
        hypothesis = 1 / (1 + np.exp(-z))
        hypothesis = np.asarray([int(element + 0.5) for element in hypothesis])
        # Update weight in the direction of the partial derivative
        weight += learning_rate * np.dot(y - hypothesis, x) / len(trainData)
        epoch += 1
        eTrain = ((hypothesis - trainLabel) ** 2).mean(axis=None)
    # Validation
    hypoTest = 1 / (1 + np.exp(-np.dot(xTest, weight)))
    hypoTest = np.asarray([int(element + 0.5) for element in hypoTest])
    eTest = ((hypoTest - testLabel) ** 2).mean(axis=None)
    return (eTrain, eTest)

def polyTrans(x):
    # change x=(1, x1, x2) into (1, x1, x2, x1^2, x2^2, x1x2, x1^3, ...)
    x_poly = np.empty((len(x), 10))
    for i in range(len(x)):
        xi = x[i][1]
        yi = x[i][2]
        new_ele = [1, xi, yi, xi ** 2, yi ** 2, xi * yi, xi ** 3, yi ** 3, xi ** 2 * yi, yi ** 2 * xi]
        x_poly[i] = np.asarray(new_ele)
    return x_poly
