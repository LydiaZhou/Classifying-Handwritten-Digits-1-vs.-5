import numpy as np
import matplotlib.pyplot as plt
import yaml
import digitsExtractor

# Load Config info
with open("settings.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

# Vairable Declaration
weights = (1, 1, 1)
pocketIndicator = data_loaded['pocket']
max_epoches = data_loaded['max_epoches']

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

    # Linear Regression
    for iter in range(max_epoches):
        x = np.asarray(trainData)
        y = np.array(weights)
        pseudoInverseX = np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x))
        wLin = np.dot(pseudoInverseX, y)
        # hypothesis = np.dot(np.array(trainData), np.array(weights))
        # loss = hypothesis - trainLabel
    print("Finished")