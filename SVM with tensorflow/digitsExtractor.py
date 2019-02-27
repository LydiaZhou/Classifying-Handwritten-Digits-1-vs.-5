import numpy as np
import yaml

# Extract the digits as matrices from the text file, leaving out the 1 and 5 only in this project
# @return dictionary with ones and fives
def extractFile(filename):
    dict = {'one':[],'others':[]}
    Outdata, label = [], []
    currentFile = open(filename)
    featureRange = [[float('inf'), -float('inf')], [float('inf'), -float('inf')]]
    for line in currentFile:
        data = str.split(line)
        digit = int(float(data[0]))
        # Seperate 1 and the other digits
        matrix = [float(pixel) for pixel in data[1:]]
        featureTuple = featureExtractor(matrix)
        # Store the min and max values for the normilization
        if featureTuple[0] > featureRange[0][1]:
            featureRange[0][1] = featureTuple[0]
        if featureTuple[0] < featureRange[0][0]:
            featureRange[0][0] = featureTuple[0]
        if featureTuple[1] > featureRange[1][1]:
            featureRange[1][1] = featureTuple[1]
        if featureTuple[1] < featureRange[1][1]:
            featureRange[1][0] = featureTuple[1]
        if digit == 1:
            dict['one'].append(featureTuple)
            Outdata.append([featureTuple[0], featureTuple[1]])
            label.append(1)
        else:
            dict['others'].append(featureTuple)
            Outdata.append([featureTuple[0], featureTuple[1]])
            label.append(0)
    return dict, Outdata, label, featureRange

# From a matrix extract two features, symmetry in both side and density
# @return symmetryValue and densityValue
def featureExtractor(matrix):
    densitySum, symmetryVerti = 0, 0
    for i in range(16):
        for j in range(16):
            densitySum += matrix[i*16+j]
            if j < 8:
                symmetryVerti += np.absolute(matrix[i*16+j]-matrix[i*16+15-j])
    return symmetryVerti, densitySum
