print('cnv_test_lstm.py: Beginning execution')

# Imports

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Constants

data_dim = 32  # Each time contains 32 pieces of tracking data
layer_dim = 32  # Number of default units in each layer
output_dim = 1  # Output dimension
timesteps = 8  # Total number of times to consider
n_classes = 1  # Binary classification, either smiling or not, talking or not, etc.


# Set up model architecture

model = Sequential()
# Input layer
model.add(LSTM(data_dim,
               return_sequences=True,
               input_shape=(timesteps, data_dim)))
# Hidden layers
model.add(LSTM(layer_dim,
               return_sequences=True))
model.add(LSTM(layer_dim))  # Returns a single vector of dimension layer_dim
# Output layer
model.add(Dense(output_dim, activation ='softmax'))
# Compile
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Initialize data into numpy arrays
predictors = np.loadtxt('C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt', skiprows=13)
labels = np.loadtxt('C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat', skiprows=1)

# Train
x_train = predictors

# Test


print('cnv_test_lstm.py: Completed execution')

'''
# Alternatively, use mean prediction metric for comparison

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy', mean_pred])
'''
