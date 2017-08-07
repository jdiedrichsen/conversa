# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
try:
    import cnv_data  # Ignore import error, import works
except ImportError:
    print('Unable to import cnv_data')


# Parameters -----------------------------------------------------------------------------------------------------------

# File params
tracking_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\test\\sequence_0110_predictors.txt'
label_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\test\\sequence_0011_labels.txt'

# Input params
TIMESTEPS = 4  # TODO: Add to params

# Layer params
DEFAULT_LAYER_WIDTH = 4  # TODO: Add to params
N_HIDDEN_LAYERS = 2  # TODO: Add to params

# Compilation params
N_EPOCHS = 1000  # TODO: Add to params
VALIDATION_SPLIT = 0.5  # TODO: Add to

# Functions
# INPUT_FUNCTION = 'relu'  # TODO: Add to params
# HIDDEN_ACT_FUNC = 'relu'  # TODO: Add to params
OUTPUT_FUNCTION = 'softmax'  # TODO: Add to params


# Load data ------------------------------------------------------------------------------------------------------------

# Load files
try:
    predictors = np.genfromtxt(tracking_file)
    labels = np.genfromtxt(label_file)
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

