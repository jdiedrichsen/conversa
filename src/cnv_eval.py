from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.models import Sequential()
from keras.layers import LSTM
try:
    import cnv_data  # Ignore import error, import works
except ImportError:
    print('Unable to import cnv_data')

TRACKING_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
LABEL_FILE = 'C:\\Users\\Shayn\\Documents\\Work\\AI Research\\conversa\\data\\labels\\p2024cam1.dat'

# TODO: Add verbose flag and vprint function


def evaluate(models, predictors, labels, n_splits=5, n_epochs=100, batch_sz=10):
    kfold = StratifiedKFold(n_splits=n_splits)
    accuracies = []
    for train, test in kfold.split(predictors, labels):
        print('Train: ' + str(train))
        print('Test: ' + str(test))
        for model in models:
            # Fit
            model.fit(predictors[train], labels[train], epochs=n_epochs, batch_size=batch_sz, verbose=0)
            # Evaluate
            scores = model.evaluate(predictors[test], labels[test], verbose=0)
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

# Load files
try:
    (predictors, labels) = (cnv_data.load(TRACKING_FILE, LABEL_FILE))
except IOError:
    print('Failed to open files')
print('Loaded files')

model = Sequential()
model.add()

# TODO: mean prediction and LDA


