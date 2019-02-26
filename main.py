import numpy as np
import matplotlib.pyplot as plt
import yaml
import digitsExtractor

# Load Config info
with open("settings.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

# Vairable Declaration
perceptronIndicator = data_loaded['perceptron']
linear1Indicator = data_loaded['linearRegMinSquare']
linear2Indicator = data_loaded['linearRegGradient']
max_epoches = data_loaded['max_epoches']
learning_rate = data_loaded['learning_rate']

# Plotting function for generating the virsualization graph
def plot(data):
    ones_sym = [sym for (sym, density) in data[1]]
    ones_den = [density for (sym, density) in data[1]]
    fives_sym = [sym for (sym, density) in data[5]]
    fives_den = [density for (sym, density) in data[5]]
    plt.plot(ones_sym,ones_den,'ro')
    plt.plot(fives_sym, fives_den, 'bo')
    plt.show()

if __name__=="__main__":
    # Extract the data from source text file first
    # Dict for plotting; Data for training(without label)
    testDict, testData, testLabel = digitsExtractor.extractFile("ZipDigits.test")
    trainDict, trainData, trainLabel = digitsExtractor.extractFile("ZipDigits.train")
    # Plot
    plot(trainDict)

    if (linear1Indicator):
        # Linear Regression with least square method
        x = np.asarray(trainData)
        y = np.array(trainLabel)
        pseudoInverseX = np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x))
        wLin = np.dot(pseudoInverseX, y)
        hypothesis = np.dot(x, wLin)
        # Convert float result in hypothesis into 1 or 0
        hypothesis = np.asarray([int(element+0.5) for element in hypothesis])
        eTrain = ((hypothesis - trainLabel)**2).mean(axis=None)
        print("==========Linear Regression============")
        print("The training error is %.5f" %eTrain)
        # print("The balanced weights is %.5f" %wLin)

    if (perceptronIndicator):
        # PLA(Perceptron) with pocket algorithm as improvement
        weights = (1, 1)
        minEtrain = 100
        minWeight = weights
        for iter in range(max_epoches):
            for i in range(len(trainData)):
                # If the x1*w1+x2*w2>0, the number should be five(label = 0); If the x1*w1+x2*w2<0, label =1
                # else, re-balance the weights
                calculatedLabel = trainData[i][1]*weights[0]+trainData[i][2]*weights[1]
                if ((calculatedLabel > 0 and trainLabel[i] == 0) or (calculatedLabel < 0 and trainLabel[i] == 1)):
                    continue
                else:
                    aLabel = calculatedLabel
                    aArray = np.asarray(trainData[i][1:])
                    a = learning_rate * calculatedLabel * np.asarray(trainData[i][1:])/(trainData[i][1]**2+trainData[i][2]**2)
                    weights -= a
                    # hypothesis = x1*w1+x2*w2
                    hypothesis = np.dot(np.asarray(trainData)[:, 1:], np.asarray(weights))
                    hypothesis = np.asarray([(0 if element > 0 else 1) for element in hypothesis])
                    eTrain = ((hypothesis - trainLabel) ** 2).mean(axis=None)
                    if eTrain < minEtrain:
                        minEtrain = eTrain
                        minWeight = weights
        print("==========Perceptron with Pocket============")
        print("The training error is %.5f" %minEtrain)
        # print("The balanced weights is %.5f" %minWeight)

    if (linear2Indicator):
        # Linear Regression with Graident Descent
        tolerance = 0.2
        difference = 1
        epoch = 0
        x = np.asarray(trainData)
        y = np.array(trainLabel)
        weight = np.zeros(3)
        while(difference > tolerance and epoch < max_epoches):
            hypothesis = np.dot(x, weight)
            difference = np.absolute(np.sum(y - hypothesis))
            # Update weight in the direction of the partial derivative
            step1 = len(trainData)
            step2 = y - hypothesis
            step3 = x
            step4 = np.dot(y - hypothesis, x)
            weight += learning_rate*np.dot(y - hypothesis, x)/len(trainData)/1000
            epoch += 1
            eTrain = ((hypothesis - trainLabel) ** 2).mean(axis=None)
            print(difference)
        print("==========Linear Regression with Gradient Descent============")
        print("The training error is %.5f" % eTrain)







