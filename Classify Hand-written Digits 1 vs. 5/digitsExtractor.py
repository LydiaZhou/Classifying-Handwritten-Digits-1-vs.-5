import numpy as np

# Extract the digits as matrices from the text file, leaving out the 1 and 5 only in this project
# @return dictionary with ones and fives
def extractFile(filename):
    dict = {1:[],5:[]}
    Outdata, label = [], []
    currentFile = open(filename)
    for line in currentFile:
        data = str.split(line)
        digit = int(float(data[0]))
        # Leave out 1 and 5 only
        if digit != 1 and digit != 5:
            continue
        matrix = [float(pixel) for pixel in data[1:]]
        featureTuple = featureExtractor(matrix)
        if digit == 1:
            dict[1].append(featureTuple)
            Outdata.append([1, featureTuple[0], featureTuple[1]])
            label.append(1)
        else:
            dict[5].append(featureTuple)
            Outdata.append([1, featureTuple[0], featureTuple[1]])
            label.append(0)
    return dict, Outdata, label

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
