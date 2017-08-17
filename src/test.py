print('Beginning test script')

# Imports --------------------------------------------------------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
try:
    import cnv_data, cnv_eval
except ImportError:
    print('Unable to import cnv_data')
from tabulate import tabulate

# Parameters -----------------------------------------------------------------------------------------------------------

TRACKING_FILE = '..\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
LABEL_FILE = '..\\data\\labels\\p2024cam1.dat'

TIMESTEPS = 30  # Keep in mind that n_seqs = int(seq_len / TIMESTEPS)
BATCH_SZ = 10  # Optionally can set batch_size in fitting/evaluation to number of sequences (n_seqs for all sequences)
N_EPOCHS = 10
# VALIDATION_SPLIT = 0.2

# Layer params
DEFAULT_LAYER_WIDTH = 32
N_HIDDEN_LAYERS = 3
# Functions
# INPUT_FUNCTION = 'relu'
# HIDDEN_ACT_FUNC = 'relu'
OUTPUT_FUNCTION = 'softmax'


# Load data ------------------------------------------------------------------------------------------------------------

# Load files
predictors, labels = None, None
try:
    (predictors, labels) = (cnv_data.load(TRACKING_FILE, LABEL_FILE, behaviour_fields={'smile'}, structured=True))
    # TODO: Figure out how to do this with structred numpy arrays - can implement diff in cnv_eval
except IOError:
    print('Failed to open files')
print('Loaded files')

# print(predictors.shape)
# print(labels.shape)

predictors = cnv_data.to_subseqs(predictors, TIMESTEPS)
labels = cnv_data.to_subseqs(labels, TIMESTEPS)
print('Split data into subsequences')

# print(predictors.shape)
# print(labels.shape)


# Set up models --------------------------------------------------------------------------------------------------------

# First index of shape is the number of subsequences, second is length of subsequences, third is dimensions of data
input_dim = predictors.shape[2]
output_dim = labels.shape[2]

spec_model = Sequential()
# Input layer
spec_model.add(LSTM(DEFAULT_LAYER_WIDTH,
                    return_sequences=True,
                    input_shape=(TIMESTEPS, input_dim)))
# Hidden layer(s)
for i in range(0, N_HIDDEN_LAYERS):
    spec_model.add(LSTM(DEFAULT_LAYER_WIDTH,
                        return_sequences=True))
# Output layer
spec_model.add(LSTM(output_dim,
                    return_sequences=True,
                    activation=OUTPUT_FUNCTION))
# Compile
print(spec_model.summary())
spec_model.compile(optimizer='rmsprop',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

models = [spec_model]

# Train
# print('Training')
# spec_model.fit(train_predictors, train_labels,
#           batch_size=BATCH_SZ,
#           epochs=N_EPOCHS,
#           validation_split=VALIDATION_SPLIT,
#           verbose=1)
# # Can also use batch_size=train_predictors.shape[0]

# # Evaluate
# print('Evaluating')
# loss, acc = spec_model.evaluate(test_predictors, test_labels,
#                      batch_size=test_predictors.shape[0],
#                      verbose=1)  # Accuracy is at index 1, loss at index 0
# # Can also use batch_size=test_predictors.shape[0]
# print('\n\bAccuracy: ' + str(acc))


# Testing folds --------------------------------------------------------------------------------------------------------
#
# # Check exclusivitiy of timestamps
# n_folds = 5
# folds = cnv_eval.k_fold(predictors, labels, n_folds=5)
# for fold_no in range(0, len(folds)):
#     print('Checking fold number: ' + str(fold_no+1))
#     fold = folds[fold_no]
#     train_data, test_data = fold
#     (train_predictors, train_labels) = train_data
#     (test_predictors, test_labels) = test_data
#     train_timestamps = np.unique(train_predictors['timestamp'])
#     test_timestamps = np.unique(test_predictors['timestamp'])
#     exclusive = not np.any(np.isin(train_timestamps, test_timestamps))
#     # Can also assert exclusive
#     print('Fold is exclusive') if exclusive else print('TEST FAILED: FOLD IS NOT EXCLUSIVE')
#
# # Outputs via field indexing
# n_folds = 5
# folds = cnv_eval.k_fold(predictors, labels, n_folds=5)
# for fold_no in range(0, 1):
#     print('Fold number:\t' + str(fold_no))
#     fold = folds[fold_no]
#     train_data, test_data = fold
#     (train_predictors, train_labels) = train_data
#     (test_predictors, test_labels) = test_data
#     for i in range(0, 5):
#         print('Test data:\t' + str((test_predictors['timestamp'][i]*30)))
#         for j in range(0, n_folds-1):
#             print('Train data:\t' + str((train_predictors['timestamp'][i*(n_folds-1)+j]*30)))
#
# # Outputs via index
# n_folds = 5
# folds = cnv_eval.k_fold(predictors, labels, n_folds=5)
# for (train_data, test_data) in folds:
#     (train_predictors, train_labels) = train_data
#     (test_predictors, test_labels) = test_data
#     for i in range(0, 3):
#         print('Test data:\n' + str(test_predictors[i]))
#         for j in range(0, n_folds - 1):
#             print('Train data:\n' + str((train_predictors[i * (n_folds - 1) + j])))
#
#
# Testing model evalution ----------------------------------------------------------------------------------------------

eval_results = cnv_eval.eval_models(models, predictors, labels, verbose=0)

print(tabulate(eval_results, headers='keys'))




# End ------------------------------------------------------------------------------------------------------------------
#
# print('cnv_test_lstm.py: Completed execution')
