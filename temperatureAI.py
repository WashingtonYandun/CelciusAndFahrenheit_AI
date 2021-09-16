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

#model with Adam
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.09),
    loss='mean_squared_error' #'better few big errors' 
)

#review the training
plt.xlabel("Cicle")
plt.ylabel("Loss Mag")
plt.plot(trainingHistorial.history["loss"])

#see the training
print("Training...")
trainingHistorial = model.fit(celsius, fahrenheit, epochs = 400, verbose=True) #Verbose is not necesary could be false 
print("Already Trained C:!")

print("IA predict!")
result = model.predict([10.0]) #try it
print("Result... " + str(result) + " fahrenheit!")
