import numpy as np
import matplotlib.pyplot as plt
import yaml
import digitsExtractor

# Load Config info
with open("settings.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

# Vairable Declaration
perceptronIndicator = data_loaded['perceptron']
linearIndicator = data_loaded['linearReg']
logisticIndicator = data_loaded['logisticReg']
max_epoches = data_loaded['max_epoches']
learning_rate = data_loaded['learning_rate']
polyIndicator = data_loaded['3rdOrder']
weight_poly = 0
x_poly = 0
weight = 0

# Plotting function for generating the virsualization graph
def plot(data):
    ones_sym = [sym for (sym, density) in data[1]]
    ones_den = [density for (sym, density) in data[1]]
    fives_sym = [sym for (sym, density) in data[5]]
    fives_den = [density for (sym, density) in data[5]]
    plt.plot(ones_sym,ones_den,'ro')
    plt.plot(fives_sym, fives_den, 'bo')
    plt.show()

# Linear Regression with min square eq
def linearReg(x, weight):
    # Linear Regression with least square method
    pseudoInverseX = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x))
    wLin = np.dot(pseudoInverseX, y)
    hypothesis = np.dot(x, wLin)
    # Convert float result in hypothesis into 1 or 0
    hypothesis = np.asarray([int(element + 0.5) for element in hypothesis])
    eTrain = ((hypothesis - trainLabel) ** 2).mean(axis=None)
    return eTrain

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

def logisticReg(x, weight):
    # Logistic Regression with Graident Descent
    tolerance = 0.02
    epoch = 0
    eTrain = 1
    while (eTrain > tolerance and epoch < max_epoches):
        z = np.dot(x, weight)
        hypothesis = 1 / (1 + np.exp(-z))
        # Update weight in the direction of the partial derivative
        weight += learning_rate * np.dot(y - hypothesis, x) / len(trainData)
        epoch += 1
        eTrain = ((hypothesis - trainLabel) ** 2).mean(axis=None)
    return eTrain


if __name__=="__main__":
    # Extract the data from source text file first
    # Dict for plotting; Data for training(without label)
    testDict, testData, testLabel = digitsExtractor.extractFile("ZipDigits.test")
    trainDict, trainData, trainLabel = digitsExtractor.extractFile("ZipDigits.train")
    # Plot
    plot(trainDict)
    weight = np.zeros(3)

    if (polyIndicator):
        # change x=(1, x1, x2) into (1, x1, x2, x1^2, x2^2, x1x2, x1^3, ...)
        x = np.asarray(trainData)
        y = np.array(trainLabel)
        x_poly = np.empty((len(x),10))

        for i in range(len(x)):
            xi = x[i][1]
            yi = x[i][2]
            new_ele = [1, xi, yi, xi**2, yi**2, xi*yi, xi**3, yi**3, xi**2*yi, yi**2*xi]
            x_poly[i] = np.asarray(new_ele)
        weight_poly = np.zeros(10)

    if (linearIndicator):
        print("==========Linear Regression============")
        if (polyIndicator):
            eTrain_poly = linearReg(x_poly, weight_poly)
            print("The training error with 3rd order polynormial is %.5f" %eTrain_poly)
        eTrain = linearReg(x, weight)
        print("The training error is %.5f" % eTrain)

    if (perceptronIndicator):
        print("==========Perceptron with Pocket============")
        perceptron()

    if (logisticIndicator):
        print("==========Linear Regression with Gradient Descent============")
        if (polyIndicator):
            eTrain_poly = logisticReg(x_poly, weight_poly)
            print("The training error with 3rd order polynormial is %.5f" %eTrain_poly)
        eTrain = logisticReg(x, weight)
        print("The training error is %.5f" % eTrain)







