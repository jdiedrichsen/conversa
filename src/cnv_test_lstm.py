# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
import random  # TODO: RM
try:
    import cnv_data, cnv_eval
except ImportError:
    print('Unable to import cnv_data')


# Parameters -----------------------------------------------------------------------------------------------------------

TRACKING_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
LABEL_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'

TIMESTEPS = 30  # Keep in mind that n_seqs = int(seq_len / TIMESTEPS)
BATCH_SZ = 10  # Optionally can set batch_size in fitting/evaluation to number of sequences (n_seqs for all sequences)
N_EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Layer params
DEFAULT_LAYER_WIDTH = 1
N_HIDDEN_LAYERS = 1
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

# Set up spec_model ----------------------------------------------------------------------------------------------------

spec_model = Sequential()
# Input layer
spec_model.add(LSTM(DEFAULT_LAYER_WIDTH,
                    return_sequences=True,
                    input_shape=(TIMESTEPS, input_dim)))
# Hidden layer(s)
for i in range(0, N_HIDDEN_LAYERS):
    spec_model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
# Output layer
spec_model.add(LSTM(output_dim,
                    return_sequences=True,
                    activation=OUTPUT_FUNCTION))
# Compile
print(spec_model.summary())
spec_model.compile(optimizer='rmsprop',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])


# TODO: RM
def set_avg(d):
    for i in range(0, len(d)):
        d[i] = d[i] - random.uniform(0, 0.01)
    return d

n_test_seqs = int(n_seqs / 5)
n_train_seqs = n_seqs - n_test_seqs

# train_predictors = predictors[:n_train_seqs]
# train_labels = labels[:n_train_seqs]
# test_predictors = predictors[n_train_seqs:]
# test_labels = labels[n_train_seqs:]

# indices = cnv_eval.k_fold(predictors, labels, 5)
#
# (train_predictors, test_predictors, train_labels, test_labels) = indices[0]
#
# print('Predictor shape: ' + str(predictors.shape))
# print('Train predictor shape: ' + str(train_predictors.shape))
# print('Test predictor shape: ' + str(test_predictors.shape))
#
# print('Training timestamp range: [' + str(train_predictors['timestamp'][0][0][0]) + ', ' +
#       str(train_predictors['timestamp'][-1][-1][0]) + ']')
# print('Testing timestamp range: [' + str(test_predictors['timestamp'][0][0][0]) + ', ' +
#       str(test_predictors['timestamp'][-1][-1][0]) + ']')


# Evaluate -------------------------------------------------------------------------------------------------------------

test_models = []

test_models.append(spec_model)

spec_model_2 = Sequential()
# Input layer
spec_model_2.add(LSTM(DEFAULT_LAYER_WIDTH,
                    return_sequences=True,
                    input_shape=(TIMESTEPS, input_dim)))
# # Hidden layer(s)
# for i in range(0, N_HIDDEN_LAYERS):
#     spec_model_2.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
# Output layer
# spec_model_2.add(LSTM(output_dim,
#                     return_sequences=True,
#                     activation=OUTPUT_FUNCTION))
# Compile
print(spec_model_2.summary())
spec_model_2.compile(optimizer='rmsprop',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

test_models.append(spec_model_2)

n_spl = 5

scores = cnv_eval.eval_models(test_models, predictors, labels,
                              n_splits=n_spl,
                              train_n_epochs=10,
                              train_batch_sz=10,
                              test_n_epochs=1,
                              test_n_batch_sz=1)

# print(scores)
# for i in range(0, len(test_models)):
#     for j in range(0, n_spl):
#         print('Accuracy of model ' + str(i+1) + ' in fold ' + str(j+1) + ':\t' + str(scores[i][:][1]))


# for score in scores:
#     print(str(score))


model_accs = [scores[i][1] for i in range(0, len(scores))]
avg_accs = []
for i in range(0, len(test_models)):
    model_acc = model_accs[i:len(test_models)]
    avg_accs.append(np.average(model_accs))
avg_accs = set_avg(avg_accs)

# Header
print('\nModel\t\tAverage accuracy')
# # Entries
# for i in range(0, len(avg_accs)):
#     print('LSTM_model_' + str(i) + '\t' + str(avg_accs[i]))


# Entries
for i in range(0, len(test_models)):
    print('LSTM_model_' + str(i) + '\t' + str(scores[(i+1)*n_spl-1][1] - 0.5*scores[(i+1)*n_spl-1][0]))
    # print('LSTM_model_' + str(i) + '\t' + str(scores[(i+1)*n_spl-1]))


# print('Accuracy: '.join(str(scores[:][0])))


# print(train_predictors['timestamp'][0])
# print(test_predictors['timestamp'][0])



# Train
# print('Training')
# spec_model.fit(train_predictors, train_labels,
#           batch_size=BATCH_SZ,
#           epochs=N_EPOCHS,
#           validation_split=VALIDATION_SPLIT,
#           verbose=1)
# # Can also use batch_size=train_predictors.shape[0]
#
# # Evaluate
# print('Evaluating')
# loss, acc = spec_model.evaluate(test_predictors, test_labels,
#                      batch_size=test_predictors.shape[0],
#                      verbose=1)  # Accuracy is at index 1, loss at index 0
# # Can also use batch_size=test_predictors.shape[0]
# print('\n\bAccuracy: ' + str(acc))


# End ------------------------------------------------------------------------------------------------------------------

print('cnv_test_lstm.py: Completed execution')

