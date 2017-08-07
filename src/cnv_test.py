# Imports

# import tensorflow as tf
import numpy as np
try:
    import cnv_data  # Ignore import error, import works
except ImportError:
    print('Unable to import cnv_data')
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Embedding


# Trial on test data ===================================================================================================


tr_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\test\\sequence_0110_predictors.txt'
la_file = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\test\\sequence_0011_labels.txt'

try:
    predictors = np.genfromtxt(tr_file)
    labels = np.genfromtxt(la_file)
except IOError:
    print('Failed to open files')

print('Loaded files')

# Constants

# Input
TIMESTEPS = 4
BATCH_SIZE = 64

INPUT_DIM = 1
DEFAULT_LAYER_WIDTH = 4
OUTPUT_DIM = 1
N_EPOCHS = 1000
VALIDATION_SPLIT = 0.5

# Functions
INPUT_FUNCTION = 'relu'
HIDDEN_ACT_FUNC = 'relu'
OUTPUT_FUNCTION = 'softmax'

# Add dimensions
# Predictors need to be 3D (batch_size, timesteps, input_dimension)
predictors = cnv_data.add_dim(predictors)
predictors = cnv_data.add_dim(predictors)
# Labels need to be 2D (timesteps, output_dimension
labels = cnv_data.add_dim(labels)
labels = cnv_data.add_dim(labels)

print('Input shape:' + str(predictors.shape))
print('Output shape: ' + str(labels.shape))

# Shape into timesteps of appropriate size
predictors = np.reshape(predictors, (BATCH_SIZE, TIMESTEPS, INPUT_DIM))
labels = np.reshape(labels, (BATCH_SIZE, TIMESTEPS, OUTPUT_DIM))

# # print('Predictors ===============================')
# print(predictors[0])
# # print('Labels ===================================')
# print(labels[0])

print('Input shape:' + str(predictors.shape))
print('Output shape: ' + str(labels.shape))

# Set up model
model = Sequential()
# Input layer
model.add(LSTM(DEFAULT_LAYER_WIDTH,
               return_sequences=True,
               input_shape=(TIMESTEPS, INPUT_DIM)))
# Hidden layer(s)
for i in range(0, 1):
    model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
    # model.add(LSTM(DEFAULT_LAYER_WIDTH,
    #                return_sequences=True,
    #                activation=HIDDEN_ACT_FUNC,
    #                recurrent_activation=HIDDEN_ACT_FUNC))
# Output layer
model.add(LSTM(OUTPUT_DIM,
               return_sequences=True,
               activation=OUTPUT_FUNCTION))
# model.add(Dense(OUTPUT_DIM,
#                 activation=OUTPUT_FUNCTION))
# model.add(TimeDistributed(Dense(OUTPUT_DIM, activation=OUTPUT_FUNCTION)))
# model.add(LSTM(TIMESTEPS))
# Compile
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(predictors, labels, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_split=0.8)

print('Prediction on first five sequences:')
print(model.predict(predictors[0:1]))
