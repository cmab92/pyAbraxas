import sys
#print(sys.version)
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
#print(tf.__version__)
##
rng = pd.date_range(start='2000', periods=209, freq='M')
ts = pd.Series(np.random.uniform(-10,10,size=len(rng)), rng).cumsum()
#ts.plot(c='b', title='Time Series')

#print(ts.head(10))
TS = np.array(ts)
num_periods = 20
f_horizon = 1

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, 20, 1)
y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, 20, 1)
#print(len(x_batches))
#print(x_batches.shape)
#plt.figure()
#plt.plot(x_batches[0])

def test_data(series, forecast, num_periods):
  test_x_setup = series[-(num_periods + forecast):]
  testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
  testY = series[-(num_periods):].reshape(-1, 20, 1)
  return testX, testY

X_test, Y_test = test_data(TS, f_horizon, num_periods)
#print(X_test.shape)
#print(X_test)

tf.reset_default_graph()

num_periods = 20
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

epochs = 1000

with tf.Session() as sess:
  init.run()
  for ep in range(epochs):
    sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
    if ep % 100 == 0:
      mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
      print(ep, "\tMSE", mse)
  y_pred = sess.run(outputs, feed_dict={X: X_test})
print(y_pred)
plt.title("fc vs gt")
plt.plot(pd.Series(np.ravel(Y_test)), "b", markersize=10, label="gt")
plt.plot(pd.Series(np.ravel(y_pred)),"r", markersize=10, label="fc")
plt.legend()
plt.xlabel("")
plt.show()