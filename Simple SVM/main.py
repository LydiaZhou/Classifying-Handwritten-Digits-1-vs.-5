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
batch_size = len(trainData)

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

# Optimizer
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(svm_loss)

# Evaluation.
predicted_class = tf.sign(model_output);
correct_prediction = tf.equal(y,tf.cast(predicted_class, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Start training
init = tf.global_variables_initializer()
sess.run(init)
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(100):
    sess.run(train_step, feed_dict={trainData: x, trainLabel: y})
    check1 = sess.run(regularization_loss, feed_dict={trainData: x, trainLabel: y}) / batch_size
    check2 = sess.run(hinge_loss, feed_dict={trainData: x, trainLabel: y}) / batch_size
    if i%20 == 0:
        loss = sess.run(svm_loss, feed_dict={trainData: x, trainLabel: y}) / batch_size
        print(loss)

print("==============")
if saveMe:
   save_path = saver.save(sess, saveFile)

# Verification
learnedWeight = sess.run(w)
learnedB = sess.run(b)
hypothesis = np.dot(testData, learnedWeight) - learnedB
hypothesis = np.asarray([np.sign(ele) for ele in hypothesis])
# hypothesis = np.asarray([(1 if ele > 0 else -1) for ele in hypothesis])
eTest = ((hypothesis - testLabel) ** 2).mean(axis=None)
print("The Test set error is %.5f" % eTest)




