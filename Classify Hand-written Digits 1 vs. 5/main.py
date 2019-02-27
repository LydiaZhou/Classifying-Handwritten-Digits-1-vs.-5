import numpy as np
import matplotlib.pyplot as plt
import yaml
import digitsExtractor
import learningTools

# Load Config info
with open("settings.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

# Vairable Declaration
perceptronIndicator = data_loaded['perceptron']
linearIndicator = data_loaded['linearReg']
logisticIndicator = data_loaded['logisticReg']
# max_epoches = data_loaded['max_epoches']
# learning_rate = data_loaded['learning_rate']
polyIndicator = data_loaded['3rdOrder']
plotIndicator = data_loaded['plot']
weight_poly = 0
x_poly = 0
weight = 0
# Extract the data from source text file first
# Dict for plotting; Data for training(without label)
testDict, testData, testLabel = digitsExtractor.extractFile("ZipDigits.test")
trainDict, trainData, trainLabel = digitsExtractor.extractFile("ZipDigits.train")
weight = np.zeros(3)
x = np.asarray(trainData)
y = np.array(trainLabel)
x_test = np.asarray(testData)

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
    if plotIndicator:
        print("The scatter of the training data is shown")
        plot(trainDict)

    if polyIndicator:
        x_poly = learningTools.polyTrans(x)
        xTest_poly = learningTools.polyTrans(x_test)
        weight_poly = np.zeros(10)

    if (linearIndicator):
        print("==========Linear Regression============")
        if (polyIndicator):
            (eTrain_poly, eTest_poly) = learningTools.linearReg(x_poly, weight_poly, xTest_poly)
            print("The training error with 3rd order polynormial is %.5f" %eTrain_poly)
            print("The validation error with 3rd order polynormial is %.5f" % eTest_poly)
        (eTrain, eTest) = learningTools.linearReg(x, weight, x_test)
        print("The training error is %.5f" % eTrain)
        print("The validation error is %.5f" % eTest)

    if (perceptronIndicator):
        print("==========Perceptron with Pocket============")
        learningTools.perceptron()

    if (logisticIndicator):
        print("==========Linear Regression with Gradient Descent============")
        if (polyIndicator):
            (eTrain_poly, eTest_poly) = learningTools.logisticReg(x_poly, weight_poly, xTest_poly)
            print("The training error with 3rd order polynormial is %.5f" %eTrain_poly)
            print("The validation error with 3rd order polynormial is %.5f" % eTest_poly)
        (eTrain, eTest) = learningTools.logisticReg(x, weight, x_test)
        print("The training error is %.5f" % eTrain)
        print("The validation error is %.5f" % eTest)







