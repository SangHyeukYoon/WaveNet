import tensorflow as tf
from tensorflow.python.keras.layers import Conv1D

import numpy as np

input_shape = (1, 16000, 1)
x = tf.random.normal(input_shape)

f = Conv1D(32, 2, padding='causal', dilation_rate=2, input_shape=input_shape[1:])(x)
g = Conv1D(32, 2, padding='causal', dilation_rate=2, input_shape=input_shape[1:])(x)
z = tf.tanh(f) * tf.sigmoid(g)
res = Conv1D(32, 1, padding='same')(z)

print(res.shape)

