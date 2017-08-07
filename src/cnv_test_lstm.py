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

# File params
tracking_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
label_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'

# Input params
TIMESTEPS = 100

# Layer params
DEFAULT_LAYER_WIDTH = 4
N_HIDDEN_LAYERS = 2

# Compilation params
N_EPOCHS = 1000
VALIDATION_SPLIT = 0.5

# Functions
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
SEQ_LENGTH = predictors.shape[0]
if SEQ_LENGTH != labels.shape[0]:
    raise RuntimeError('Predictor and label length mismatch')
BATCH_SIZE = int(SEQ_LENGTH/TIMESTEPS)
INPUT_DIM = predictors.shape[1]
OUTPUT_DIM = labels.shape[1]
# Trim before reshaping into batches
new_len = BATCH_SIZE*TIMESTEPS
predictors = predictors[:new_len]
labels = labels[:new_len]
# Reshape into batches
predictors = np.reshape(predictors, (BATCH_SIZE, TIMESTEPS, INPUT_DIM))
labels = np.reshape(labels, (BATCH_SIZE, TIMESTEPS, OUTPUT_DIM))


# Set up model ---------------------------------------------------------------------------------------------------------

model = Sequential()
# Input layer
model.add(LSTM(DEFAULT_LAYER_WIDTH,
               return_sequences=True,
               input_shape=(TIMESTEPS, INPUT_DIM)))
# Hidden layer(s)
for i in range(0, N_HIDDEN_LAYERS):
    model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
# Output layer
model.add(LSTM(OUTPUT_DIM,
               return_sequences=True,
               activation=OUTPUT_FUNCTION))
# Compile
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(predictors, labels, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_split=0.8)


# End ------------------------------------------------------------------------------------------------------------------

print('cnv_test_lstm.py: Completed execution')

