import numpy as np
import tensorflow as tf
import yaml
import digitsExtractor
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()
# Load Config info
with open("settings.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

# Declare the variables
normIndicator = data_loaded['normalization']
learningRate = data_loaded['learningRate']
saveMe = data_loaded['saveMe']
saveFile = data_loaded['saveFile']

# Initialize all the variables
testDict, testData, testLabel, testRange = digitsExtractor.extractFile("ZipDigits.test")
trainDict, trainData, trainLabel, trainRange = digitsExtractor.extractFile("ZipDigits.train")

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

x = np.asarray(trainData)
y = np.asarray(trainLabel)
x_test = np.asarray(testData)
batch_size = len(testData)

# Initialize the tf feedin data
trainData = tf.placeholder(shape=[None, 2], dtype=tf.float32)
trainLabel = tf.placeholder(shape=[None, ], dtype=tf.float32)

# Create the tf variables
w = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1]))

# Define the linear model
# model_output = tf.cast(tf.clip_by_value(
#     tf.subtract(tf.matmul(trainData, w), b),
#     tf.constant([1.99]),
#     tf.constant([0.]),
# ), tf.int32)
model_output = tf.subtract(tf.matmul(trainData, w), b)

# Optimization.
regularization_loss = 0.5*tf.reduce_sum(tf.square(w))
# hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size,1], tf.int32), 1 - y*model_output));
# svm_loss = regularization_loss + tf.cast(hinge_loss, tf.float32);
hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size,1]), 1 - y*model_output));
svm_loss = regularization_loss + hinge_loss;

# Construct list of tensorflow variables to save.
lst_vars = []
for v in tf.global_variables():
    lst_vars.append(v)
# Create the Saver object.
saver = tf.train.Saver(var_list=lst_vars)

# Initialize session
with tf.Session() as sess:
   # Init variables
   sess.run(tf.global_variables_initializer())
   saver.restore(sess, saveFile)

   # Evaluate all the test inputs
   label_p = sess.run(model_output, {trainData: testData})

   hit = 0;
   miss = 0;
   for i in range(batch_size):
       if ((label_p[i] >= 0.5 and testLabel[i]==1) or (label_p[i] < 0.5 and testLabel[i] == 0)):
           hit += 1
       else:
           miss += 1

   hitrate = float(hit) / float(miss + hit)

   print("hitrate on {} test points is {}.".format(batch_size, hitrate))