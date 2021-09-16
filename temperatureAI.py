#libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#numpy arrays input and output
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#keras
layer = tf.keras.layers.Dense(units=1, input_shape=[1]) #dense, all to all this 1-1
model = tf.keras.Sequential([layer])
