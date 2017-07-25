# Imports

# Tensorflow import
import tensorflow as tf
# Keras imports
from keras.models import Sequential
from keras.layers import LSTM, Dense
# Numpy imports
import numpy as mp

# Constants

data_dim = 32  # Each time contains 32 pieces of tracking data
layer_dim = 32  # Number of units in each layer
output_dim = 1  # Output dimension
timesteps = 8  # Total number of times to consider
n_classes = 1  # Binary classification, either smiling or not, talking or not, etc.

# Set up model architecture

model = Sequential()
# Input layer
model.add(LSTM(layer_dim,
               return_sequences=True,
               input_shape=(timesteps, data_dim)))
# Hidden layers
model.add(LSTM(layer_dim,
               return_sequences=True))
model.add(LSTM(layer_dim))  # Returns a single vector of dimension layer_dim
# Output layer
model.add(Dense(output_dim, activation ='softmax'))