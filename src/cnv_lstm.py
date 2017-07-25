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
timesteps = 8  # Total number of times to consider
n_classes = 2  # Binary classification, either smiling or not, talking or not, etc.

# Set up model architecture

model = Sequential()
# Includes 32 features (e.g. smile_l, jow_open_l, etc.) but excludes timestamp
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
