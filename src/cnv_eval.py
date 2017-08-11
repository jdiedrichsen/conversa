from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
try:
    import cnv_data  # Ignore import error, import works
except ImportError:
    print('Unable to import cnv_data')

# TODO: Add verbose flags and vprint function


def evaluate(models, predictors, labels, n_splits=5, n_epochs=100, batch_sz=10):
    kfold = StratifiedKFold(n_splits=n_splits)
    accuracies = []
    print(predictors.shape)
    print(labels.shape)
    for train, test in kfold.split(predictors, labels):
        print('Train: ' + str(train))
        print('Test: ' + str(test))
        for model in models:
            # Fit
            model.fit(predictors[train], labels[train], epochs=n_epochs, batch_size=batch_sz, verbose=0)
            # Evaluate
            scores = model.evaluate(predictors[test], labels[test], batch_size=batch_sz, verbose=0)
            accuracies.append(scores)  # TODO: Set to structured numpy array, fieldnames as model names
    print('Avg acc: ' + str(np.mean(accuracies)))
    return accuracies


# def eval_2(models, predictors, labels)
#
#
#
# def k_fold(n_splits, max_index):
#     folds =
#     for i in range()

# Modified from cnv_test_lstm.py =======================================================================================

# Parameters -----------------------------------------------------------------------------------------------------------

TRACKING_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
LABEL_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'

TIMESTEPS = 30  # Keep in mind that n_seqs = int(seq_len / TIMESTEPS)
N_EPOCHS = 100
VALIDATION_SPLIT = 0.5

# Layer params
DEFAULT_LAYER_WIDTH = 4
N_HIDDEN_LAYERS = 2
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

acc = evaluate([model], predictors, labels)
print(acc)

# TODO: mean prediction and LDA (and Naive Bayes?)

print('Imported cnv_eval')
