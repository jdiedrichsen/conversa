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
# BATCH_SZ = 10  # Optionally can set batch_size in fitting/evaluation to number of sequences (n_seqs for all sequences)
# N_EPOCHS = 10
# VALIDATION_SPLIT = 0.2

# Layer params
DEFAULT_LAYER_WIDTH = 32
DEFAULT_N_HIDDEN_LAYERS = 4
# Functions
# INPUT_FUNCTION = 'relu'
HIDDEN_ACT_FUNC = 'relu'
OUTPUT_FUNCTION = 'softmax'


# Load data ------------------------------------------------------------------------------------------------------------

# # Load files
# predictors, labels = None, None
# try:
#     (predictors, labels) = (cnv_data.load(TRACKING_FILE, LABEL_FILE, behaviour_fields={'smile'}, structured=True))
# except IOError:
#     print('Failed to open files')
# print('Loaded files')
#
# # print(predictors.shape)
# # print(labels.shape)
#
# predictors = cnv_data.to_subseqs(predictors, TIMESTEPS)
# labels = cnv_data.to_subseqs(labels, TIMESTEPS)
# # print('Split data into subsequences')

# print(predictors.shape)
# print(labels.shape)


# Set up models --------------------------------------------------------------------------------------------------------

# First index of shape is the number of subsequences, second is length of subsequences, third is dimensions of data

# input_dim = predictors.shape[2]
# output_dim = labels.shape[2]
# print(input_dim)
# print(output_dim)

input_dim = 1
output_dim = 1

input_shape = (TIMESTEPS, input_dim)

# # There is a keras bug where the shape elements are converted to float, caused a tensorflow error
# print(type(TIMESTEPS))
# print(type(input_dim))

# Model 1 --------------------------------------------------------------------------------------------------------------


# Temp for convenience
# Can also set recurrent activation and dropout
def mk_LSTM_model(input_shape, layer_width, n_hidden_layers, hidden_activation, output_dim, output_func):
    mdl = Sequential()
    # Input
    mdl.add(LSTM(layer_width,
                 return_sequences=True,
                 input_shape=input_shape))
    # Hidden
    for layer_no in range(0, n_hidden_layers):
        mdl.add(LSTM(layer_width,
                     activation=hidden_activation,
                     return_sequences=True))
    # Output

        mdl.add(LSTM(output_dim,
                     return_sequences=True,
                     activation=output_func))
    # Compile
        mdl.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return mdl

models = []

models.append(
    mk_LSTM_model(
        input_shape=input_shape,
        layer_width=int(DEFAULT_LAYER_WIDTH/2),
        n_hidden_layers=int(DEFAULT_N_HIDDEN_LAYERS/2),
        hidden_activation=HIDDEN_ACT_FUNC,
        output_dim=output_dim,
        output_func=OUTPUT_FUNCTION
    )
)

models.append(
    mk_LSTM_model(
        input_shape=input_shape,
        layer_width=DEFAULT_LAYER_WIDTH,
        n_hidden_layers=DEFAULT_N_HIDDEN_LAYERS,
        hidden_activation=HIDDEN_ACT_FUNC,
        output_dim=output_dim,
        output_func=OUTPUT_FUNCTION
    )
)

models.append(
    mk_LSTM_model(
        input_shape=input_shape,
        layer_width=int(DEFAULT_LAYER_WIDTH*2),
        n_hidden_layers=int(DEFAULT_N_HIDDEN_LAYERS*2),
        hidden_activation=HIDDEN_ACT_FUNC,
        output_dim=output_dim,
        output_func=OUTPUT_FUNCTION
    )
)

# model_1 = Sequential()
# # Input layer
# model_1.add(LSTM(DEFAULT_LAYER_WIDTH,
#                  return_sequences=True,
#                  input_shape=input_shape))
# # Hidden layer(s)
# for i in range(0, DEFAULT_N_HIDDEN_LAYERS):
#     model_1.add(LSTM(DEFAULT_LAYER_WIDTH,
#                      return_sequences=True))
# # Output layer
# model_1.add(LSTM(output_dim,
#                  return_sequences=True,
#                  activation=OUTPUT_FUNCTION))
# # Compile
# print(model_1.summary())
# model_1.compile(optimizer='rmsprop',
#                 loss='binary_crossentropy',
#                 metrics=['accuracy'])

# model_2 = Sequential()
# # Input layer
# model_2.add(LSTM(int(DEFAULT_LAYER_WIDTH / 2),
#                  return_sequences=True,
#                  input_shape=input_shape))
# # Hidden layer(s)
# for i in range(0, DEFAULT_N_HIDDEN_LAYERS*2):
#     model_2.add(LSTM(int(DEFAULT_LAYER_WIDTH / 2),
#                      return_sequences=True))
# # Output layer
# model_2.add(LSTM(output_dim,
#                  return_sequences=True,
#                  activation=OUTPUT_FUNCTION))
# # Compile
# print(model_2.summary())
# model_2.compile(optimizer='rmsprop',
#                 loss='binary_crossentropy',
#                 metrics=['accuracy'])



# Train
# print('Training')
# model_1.fit(train_predictors, train_labels,
#           batch_size=BATCH_SZ,
#           epochs=N_EPOCHS,
#           validation_split=VALIDATION_SPLIT,
#           verbose=1)
# # Can also use batch_size=train_predictors.shape[0]

# # Evaluate
# print('Evaluating')
# loss, acc = model_1.evaluate(test_predictors, test_labels,
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

# small_eval_results = cnv_eval.eval_models(models, predictors, labels, verbose=0)
#
# print(tabulate(small_eval_results, headers='keys'))

for mdl in models:
    print(mdl.summary())

subjects = [
    (1001, 1),
    (1005, 1),
    # (2001, 1),
    # (2002, 1),
    # (2006, 1),
    # (2010, 1),
    # (2017, 1),
    # (2024, 1),
]

behavs = {
    'smile',
    'talk',
    # 'laugh',
}

eval_results = cnv_eval.eval_models_on_subjects(models, subjects, timesteps=TIMESTEPS, behaviours=behavs)

print(tabulate(eval_results, headers='keys'))

# # Quick test of accuracy
# cnv_eval.accuracy(
#     np.array([
#         np.array([0, 0]),
#         np.array([0, 0]),
#         np.array([0, 0]),
#         np.array([0, 0]),
#         np.array([0, 0]),
#     ]),
#     np.array([
#         np.array([0, 0]),
#         np.array([0, 0]),
#         np.array([0, 0]),
#         np.array([0, 0]),
#         np.array([0, 0]),
#     ])
# )


# End ------------------------------------------------------------------------------------------------------------------
#
# print('cnv_test_LSTM.py: Completed execution')
