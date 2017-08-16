print('Beginning test script')
try:
    import cnv_data, cnv_eval
except ImportError:
    print('Unable to import cnv_data')


TRACKING_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
LABEL_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'

predictors, labels = None, None
try:
    (predictors, labels) = (cnv_data.load(TRACKING_FILE, LABEL_FILE, {'smile'}))
except IOError:
    print('Failed to open files')


# Testing folds

n_folds = 5
folds = cnv_eval.k_fold(predictors, labels, 5)
for fold_no in range(0, len(folds)):
    print('Fold number:\t' + str(fold_no))
    fold = folds[fold_no]
    train_data, test_data = fold
    (train_predictors, train_labels) = train_data
    (test_predictors, test_labels) = test_data
    for i in range(0, 1):
        print('Test:\t' + str(test_predictors['timestamp'][i]))
        for j in range(0, 5):
            print('Train:\t' + str(train_predictors['timestamp'][i*5+j]))

# # Imports --------------------------------------------------------------------------------------------------------------
#
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM
# try:
#     import cnv_data, cnv_eval, cnv_net
# except ImportError:
#     print('Unable to import cnv_data')
#
#
# # Parameters -----------------------------------------------------------------------------------------------------------
#
# TRACKING_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
# LABEL_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'
#
# TIMESTEPS = 30  # Keep in mind that n_seqs = int(seq_len / TIMESTEPS)
# BATCH_SZ = 10  # Optionally can set batch_size in fitting/evaluation to number of sequences (n_seqs for all sequences)
# N_EPOCHS = 10
# VALIDATION_SPLIT = 0.2
#
# # Layer params
# DEFAULT_LAYER_WIDTH = 1
# N_HIDDEN_LAYERS = 1
# # Functions
# # INPUT_FUNCTION = 'relu'
# # HIDDEN_ACT_FUNC = 'relu'
# OUTPUT_FUNCTION = 'softmax'
#
#
# # Load data ------------------------------------------------------------------------------------------------------------
#
# # Load files
# predictors, labels = None, None
# try:
#     (predictors, labels) = (cnv_data.load(TRACKING_FILE, LABEL_FILE))
# except IOError:
#     print('Failed to open files')
# print('Loaded files')
# # Set constants
# seq_len = predictors.shape[0]
# if seq_len != labels.shape[0]:
#     raise RuntimeError('Predictor and label length mismatch')
# n_seqs = int(seq_len / TIMESTEPS)
# input_dim = predictors.shape[1]
# output_dim = labels.shape[1]
# # Trim before reshaping into batches
# new_len = n_seqs * TIMESTEPS
# predictors = predictors[:new_len]
# labels = labels[:new_len]
# # # Reshape into batches
# predictors = np.reshape(predictors, (n_seqs, TIMESTEPS, input_dim))
# labels = np.reshape(labels, (n_seqs, TIMESTEPS, output_dim))
# # print(predictors.shape)
# # print(labels.shape)
# # print(n_seqs)
#
# # Set up spec_model ---------------------------------------------------------------------------------------------------------
#
# spec_model = Sequential()
# # Input layer
# spec_model.add(LSTM(DEFAULT_LAYER_WIDTH,
#                     return_sequences=True,
#                     input_shape=(TIMESTEPS, input_dim)))
# # Hidden layer(s)
# for i in range(0, N_HIDDEN_LAYERS):
#     spec_model.add(LSTM(DEFAULT_LAYER_WIDTH, return_sequences=True))
# # Output layer
# spec_model.add(LSTM(output_dim,
#                     return_sequences=True,
#                     activation=OUTPUT_FUNCTION))
# # Compile
# print(spec_model.summary())
# spec_model.compile(optimizer='rmsprop',
#                    loss='binary_crossentropy',
#                    metrics=['accuracy'])
#
# # n_test_seqs = int(n_seqs / 5)
# # n_train_seqs = n_seqs - n_test_seqs
#
# test_models = []
# test_models.append(spec_model)
#
# cnv_eval.eval_models(test_models, predictors, labels)
#
# # train_predictors = predictors[:n_train_seqs]
# # train_labels = labels[:n_train_seqs]
# # test_predictors = predictors[n_train_seqs:]
# # test_labels = labels[n_train_seqs:]
#
# # indices = cnv_eval.k_fold(predictors, labels, 5)
# #
# # (train_predictors, test_predictors, train_labels, test_labels) = indices[0]
# #
# # print('Predictor shape: ' + str(predictors.shape))
# # print('Train predictor shape: ' + str(train_predictors.shape))
# # print('Test predictor shape: ' + str(test_predictors.shape))
# #
# # print('Training timestamp range: [' + str(train_predictors['timestamp'][0][0][0]) + ', ' +
# #       str(train_predictors['timestamp'][-1][-1][0]) + ']')
# # print('Testing timestamp range: [' + str(test_predictors['timestamp'][0][0][0]) + ', ' +
# #       str(test_predictors['timestamp'][-1][-1][0]) + ']')
#
#
# # print(train_predictors['timestamp'][0])
# # print(test_predictors['timestamp'][0])
#
#
# # # Train
# # print('Training')
# # spec_model.fit(train_predictors, train_labels,
# #           batch_size=BATCH_SZ,
# #           epochs=N_EPOCHS,
# #           validation_split=VALIDATION_SPLIT,
# #           verbose=1)
# # # Can also use batch_size=train_predictors.shape[0]
#
# # # Evaluate
# # print('Evaluating')
# # loss, acc = spec_model.evaluate(test_predictors, test_labels,
# #                      batch_size=test_predictors.shape[0],
# #                      verbose=1)  # Accuracy is at index 1, loss at index 0
# # # Can also use batch_size=test_predictors.shape[0]
# # print('\n\bAccuracy: ' + str(acc))
#
#
# # End ------------------------------------------------------------------------------------------------------------------
#
# print('cnv_test_lstm.py: Completed execution')
