# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
try:
    import cnv_data
except ImportError:
    print('Unable to import cnv_data')


# Parameters -----------------------------------------------------------------------------------------------------------

TRACKING_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
LABEL_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'

TIMESTEPS = 2  # Keep in mind that n_seqs = int(seq_len / TIMESTEPS)
BATCH_SZ = 10  # Optionally can set batch_size in fitting/evaluation to number of sequences (n_seqs for all sequences)
N_EPOCHS = 100
VALIDATION_SPLIT = 0.8

# Layer params
DEFAULT_LAYER_WIDTH = 1
N_HIDDEN_LAYERS = 0
# Functions
# INPUT_FUNCTION = 'relu'
# HIDDEN_ACT_FUNC = 'relu'
OUTPUT_FUNCTION = 'softmax'


# Load data ------------------------------------------------------------------------------------------------------------

# Load files
predictors, labels = None, None
try:
    (predictors, labels) = (cnv_data.load(TRACKING_FILE, LABEL_FILE))
except IOError:
    print('Failed to open files')
print('Loaded files')
# Set constants
seq_len = predictors.shape[0]
if seq_len != labels.shape[0]:
    raise RuntimeError('Predictor and label length mismatch')
n_seqs = int(seq_len / TIMESTEPS)
input_dim = predictors.shape[1]
output_dim = labels.shape[1]
# Trim before reshaping into batches
new_len = n_seqs * TIMESTEPS
predictors = predictors[:new_len]
labels = labels[:new_len]
# Reshape into batches
predictors = np.reshape(predictors, (n_seqs, TIMESTEPS, input_dim))
labels = np.reshape(labels, (n_seqs, TIMESTEPS, output_dim))
# print(predictors.shape)
# print(labels.shape)
# print(n_seqs)

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

n_train_seqs = int(n_seqs / 2)
# n_test_seqs = n_seqs - n_train_seqs

train_predictors = predictors[n_train_seqs:]
train_labels = labels[n_train_seqs:]
test_predictors = predictors[:n_train_seqs]
test_labels = labels[:n_train_seqs]
print('predictors.shape: ' + str(predictors.shape))
print('train_predictors.shape: ' + str(train_predictors.shape))
print('test_predictors.shape: ' + str(test_predictors.shape))
# print(train_predictors['timestamp'][0])
# print(test_predictors['timestamp'][0])

# Train
print('Training')
model.fit(train_predictors, train_labels,
          batch_size=BATCH_SZ,
          epochs=N_EPOCHS,
          validation_split=VALIDATION_SPLIT, verbose=1)
# Can also use batch_size=train_predictors.shape[0]

# Evaluate
print('Evaluating')
acc = model.evaluate(test_predictors, test_labels,
                     batch_size=BATCH_SZ,
                     verbose=1)[1]  # Accuracy is at index 1, loss at index 0
# Can also use batch_size=test_predictors.shape[0]
print('\nAccuracy: ' + str(acc))


# End ------------------------------------------------------------------------------------------------------------------

print('cnv_test_lstm.py: Completed execution')
