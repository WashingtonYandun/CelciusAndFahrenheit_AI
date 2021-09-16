#libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#numpy arrays input and output
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#keras
layer = tf.keras.layers.Dense(units=1, input_shape=[1]) #dense, all to all this 1 - 1 
model = tf.keras.Sequential([layer])

#hidden layers
hiddenL1 = tf.keras.layers.Dense(units=5, input_shape=[1]) #five each one
hiddenL2 = tf.keras.layers.Dense(units=5)
finalOutput = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([hiddenL1, hiddenL2, finalOutput])

#model with Adam
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.09),
    loss='mean_squared_error' #'better few big errors' 
)

#see the training
print("Training...")
trainingHistorial = model.fit(celsius, fahrenheit, epochs = 400, verbose=False) #Verbose is not necesary could be false 
print("Already Trained C:!")

#review the training
plt.xlabel("Cicle")
plt.ylabel("Loss Mag")
plt.plot(trainingHistorial.history["loss"])

#result
print("IA predict!")
result = model.predict([15.0]) #try it
print("Result... " + str(result) + " fahrenheit!")
