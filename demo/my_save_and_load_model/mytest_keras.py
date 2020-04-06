import tensorflow as tf
from tensorflow import keras
# from tensorflow.


model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

#1
# model.compile(loss='mean_squared_error', optimizer='sgd')

#2
from keras import losses
model.compile(loss=losses.mean_squared_error, optimizer='sgd')
model.compile(loss=tf.losses.mean_squared_error, optimizer='sgd')
