# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
try:
    import cnv_data  # Ignore import error, import works
except ImportError:
    print('Unable to import cnv_data')


# Parameters -----------------------------------------------------------------------------------------------------------

 # TODO: Add to params

# Maccro params
tracking_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
label_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'
TIMESTEPS = 150
N_EPOCHS = 100
VALIDATION_SPLIT = 0.5

# Layer params
DEFAULT_LAYER_WIDTH = 32
N_HIDDEN_LAYERS = 8
# Functions, TODO: change to array of layer properties
# INPUT_FUNCTION = 'relu'
# HIDDEN_ACT_FUNC = 'relu'
OUTPUT_FUNCTION = 'softmax'


# Load data ------------------------------------------------------------------------------------------------------------

# Load files
try:
    (predictors, labels) = (cnv_data.load(tracking_file, label_file))
except IOError:
    print('Failed to open files')
print('Loaded files')
# Set constants
seq_len = predictors.shape[0]
if seq_len != labels.shape[0]:
    raise RuntimeError('Predictor and label length mismatch')
batch_sz = int(seq_len / TIMESTEPS)
input_dim = predictors.shape[1]
output_dim = labels.shape[1]
# Trim before reshaping into batches
new_len = batch_sz * TIMESTEPS
predictors = predictors[:new_len]
labels = labels[:new_len]
# Reshape into batches
predictors = np.reshape(predictors, (batch_sz, TIMESTEPS, input_dim))
labels = np.reshape(labels, (batch_sz, TIMESTEPS, output_dim))


# Set up model ---------------------------------------------------------------------------------------------------------

model = Sequential()
# Input layer
model.add(LSTM(DEFAULT_LAYER_WIDTH,
               return_sequences=True,
               input_shape=(TIMESTEPS, input_dim)))
# Hidden layer(s)
for i in range(0, N_HIDDEN_LAYERS):
    model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
# Output layer
model.add(LSTM(output_dim,
               return_sequences=True,
               activation=OUTPUT_FUNCTION))
# Compile
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(predictors, labels, batch_size=batch_sz, epochs=N_EPOCHS, validation_split=0.8)


# End ------------------------------------------------------------------------------------------------------------------

print('cnv_test_lstm.py: Completed execution')

