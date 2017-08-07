# Imports

# import tensorflow as tf
import numpy as np
try:
    import cnv_data  # Ignore import error, import works
except ImportError:
    print('Unable to import cnv_data')
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Trial on test data ===================================================================================================


tr_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\test\\sequence_0011_labels.txt'
la_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\test\\sequence_0110_predictors.txt'

try:
    predictors = np.genfromtxt(tr_file)
    labels = np.genfromtxt(la_file)
except IOError:
    print('Failed to open files')

print('Loaded files')

# Add dimensions
# Predictors need to be 3D (batch_size, )
predictors = cnv_data.add_dim(predictors)
predictors = cnv_data.add_dim(predictors)

labels = cnv_data.add_dim(labels)

# print('Input shapes:' + str(predictors.shape) + ', ' + str(labels.shape))

# Set up model

# Constants
TIMESTEPS = 16
INPUT_DIM = 16
DEFAULT_LAYER_WIDTH = 16
OUTPUT_DIM = 1
BATCH_SIZE = 10
N_EPOCHS = 10

model = Sequential()
# Input layer
model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True, input_shape=(TIMESTEPS, INPUT_DIM)))
# Hidden layer(s)
model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
# Output layers
model.add(LSTM(DEFAULT_LAYER_WIDTH))
model.add(Dense(OUTPUT_DIM))
# Compile
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Train
model.fit(predictors, labels, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_split=0.5)

exit()  # temp


# Trial on tracking data ===============================================================================================


# Constants

# Temp
tracking_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
label_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'

# Initialize data into numpy arrays
predictors, labels = cnv_data.load(tracking_file, label_file)

predictors = cnv_data.destructure(predictors)
labels = cnv_data.destructure(labels)

in_dim = 32  # Each time contains 32 pieces of tracking data
n_in_features = 35
layer_dim = 32  # Number of default units in each layer
output_dim = 1  # Output dimension
timesteps = 16  # Total number of times to consider
n_classes = 1  # Binary classification, either smiling or not, talking or not, etc.
test_ratio = 0.2  # Proportion of the data to use for cross-validation
n_epochs = 16
batch_sz = 1
print(predictors.shape)
in_shape = predictors.shape[3:]

# Set up model architecture

model = Sequential()
# Input layer
model.add(LSTM(in_dim,
               return_sequences=True,
               input_shape=(None, n_in_features)))
# Hidden layers
model.add(LSTM(layer_dim,
               return_sequences=True))
model.add(LSTM(layer_dim))
# Output layer
model.add(Dense(output_dim, activation ='softmax'))
# Compile
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])  # See bottom of file for comparing against mean prediction

# Fit and test
model.fit(predictors, labels, batch_size=batch_sz, epochs=n_epochs, validation_split=test_ratio)


# End ==================================================================================================================


print('cnv_test_lstm.py: Completed execution')
exit()  # To ensure code in next section does not run, remove line later


# Code snippets ========================================================================================================


'''
# Use mean prediction metric for comparison

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy', mean_pred])
'''


'''
# Split into training and testing sets manually

# Set indices
n_samples = predictors.shape[2]
test_n_samples = test_ratio * n_samples
train_n_samples = n_samples - test_n_samples

train_start = 0
train_end = train_n_samples
test_start = train_end  # Last index in range of numpy array is exclusive
test_end = test_start+test_n_samples

assert test_end == n_samples, 'cnv_test_lstm.py: test set does not end at end of data set'

# Set test and train predictors and labels
x_train = predictors[:][train_start:train_end]
y_train = labels[:][train_start:train_end]
x_test = predictors[:][test_start:test_end]
y_test = labels[:][test_start:test_end]
'''
