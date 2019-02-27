import numpy as np
import tensorflow as tf
from sklearn import datasets
import yaml
import digitsExtractor

sess = tf.Session()
# Load Config info
with open("settings.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

# Declare the variables
normIndicator = data_loaded['normalization']
learningRate = data_loaded['learningRate']

# Initialize all the variables
testDict, testData, testLabel, testRange = digitsExtractor.extractFile("ZipDigits.test")
trainDict, trainData, trainLabel, trainRange = digitsExtractor.extractFile("ZipDigits.train")
x = np.asarray(trainData)
y = np.array(trainLabel)
x_test = np.asarray(testData)
batch_size = len(trainData)

if normIndicator:
    # the minimum and the maximum values of the features
    minFeature1 = min(testRange[0][0], trainRange[0][0])
    maxFeature1 = max(testRange[0][1], trainRange[0][1])
    minFeature2 = min(testRange[1][0], trainRange[1][0])
    maxFeature2 = max(testRange[1][1], trainRange[1][1])
    # calculate the re-scale and the shift value
    scale1 = 2 / (maxFeature1 - minFeature1)
    shift1 = - (maxFeature1 - minFeature1)/2 - minFeature1
    scale2 = 2 / (maxFeature2 - minFeature2)
    shift2 = - (maxFeature2 - minFeature2)/2 - minFeature2
    # normalize the training and testing data
    for i in range(len(testData)):
        testData[i] = ((testData[i][0] + shift1) * scale1, (testData[i][1] + shift2) * scale2)
    for i in range(len(trainData)):
        trainData[i] = ((trainData[i][0] + shift1) * scale1, (trainData[i][1] + shift2) * scale2)

# Initialize the tf feedin data
trainData = tf.placeholder(shape=[None, 2], dtype=tf.float32)
trainLabel = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create the tf variables
w = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Define the linear model
model_output = tf.subtract(tf.matmul(trainData, w), b)

# loss function and learning rate
alpha = tf.constant([learningRate])
l2_norm = tf.reduce_sum(tf.square(w))
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, trainLabel))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# training settings /optimizer
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_step = optimizer.minimize(loss)

# Start training
init = tf.global_variables_initializer()
sess.run(init)
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(20000):
    rand_index = np.random.choice(len(trainData), size=batch_size)
    rand_x = trainData[rand_index]
    rand_y = np.transpose([trainLabel[rand_index]])
    sess.run(train_step, feed_dict={trainData: rand_x, trainLabel: rand_y})







