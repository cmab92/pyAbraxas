import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from abraxasOne.loadAndClean import loadAndClean
from gaussFilter import gaussFilter
#from sliceAndWindow import sliceAndWindowV2 as sliceAndWindow

TSAMPLE = 0.0165
irData, forceData, quatData, linAccData, angVecData = loadAndClean("_20185161226.txt", 10, 4, tSample = TSAMPLE, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/")
#plt.plot(irData[0][1000:6000,1])
#plt.show()
##
ts = irData[1][1000:5011,1]
ts_f = gaussFilter(x=1, y=ts, AMP=1, MEAN=0, SIGMA=0.15)
#plt.plot(ts)
#plt.plot(ts_f)
#plt.show()
TS = np.array(ts_f)

num_periods = 102
f_horizon = 20

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)
y_data = TS[f_horizon:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)
print(len(x_batches))
print(x_batches.shape)
#plt.figure()
#plt.plot(x_batches[0][::,0])
#plt.plot(y_batches[0][::,0])
#plt.show()


#for i in range(len(x_batches)):
#    plt.figure(1)
#    plt.plot(np.linspace(i*len(x_batches[i]), (i+1)*len(x_batches[i]),len(x_batches[i])),x_batches[i],'b')
#plt.show()

def test_data(series, forecast, num_periods):
  test_x_setup = series[-(num_periods + forecast):]
  testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
  testY = series[-(num_periods):].reshape(-1, num_periods, 1)
  return testX, testY

X_test, Y_test = test_data(TS, f_horizon, num_periods)

#plt.figure()
#plt.plot(X_test.reshape(-1),'b')
#plt.plot(Y_test.reshape(-1),'g')
#plt.show()
#print(X_test.shape)
#print(X_test)

tf.reset_default_graph()

inputs = 1
hidden = 100
output = 1
X = tf.placeholder(tf.float32, shape=[None, num_periods, inputs])
y = tf.placeholder(tf.float32, shape=[None, num_periods, inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=X, dtype=tf.float32)

learning_rate = 0.001

stacked_rnn_output = tf.reshape(tensor=rnn_output, shape=[-1, hidden])
stacked_outputs = tf.layers.dense(inputs=stacked_rnn_output, units=output)
outputs = tf.reshape(tensor=stacked_outputs, shape=[-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 1500

with tf.Session() as sess:
  init.run()
  prec = 10
  for ep in range(epochs):
    sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
    mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
    if ep % 50 == 0:
      mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
      print(ep, "\tMSE", mse)
    if mse<prec:
      prec -= 1
      y_pred = sess.run(outputs, feed_dict={X: X_test})
      plt.title("fc vs gt")
      plt.plot(pd.Series(np.ravel(Y_test)), "b", markersize=10, label="actual values")
      plt.plot(pd.Series(np.ravel(y_pred)),"r", markersize=10, label="forecast")
      plt.legend()
      plt.xlabel("")
      plt.show()
  mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
  print(ep, "\tMSE", mse)
  y_pred = sess.run(outputs, feed_dict={X: X_test})
plt.title("fc vs gt")
plt.plot(pd.Series(np.ravel(Y_test)), "b", markersize=10, label="actual values")
plt.plot(pd.Series(np.ravel(y_pred)),"r", markersize=10, label="forecast")
plt.legend()
plt.xlabel("")
plt.show()