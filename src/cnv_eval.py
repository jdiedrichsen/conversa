from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
try:
    import cnv_data  # Ignore import error, import works
except ImportError:
    print('Unable to import cnv_data')

# TODO: Add verbose flags and vprint function


def eval_models(models, predictors, labels, n_splits=5, n_epochs=100, batch_sz=10):
    kfold = StratifiedKFold(n_splits=n_splits)
    accuracies = []
    print(predictors.shape)
    print(labels.shape)
    for train_data, test_data in kfold.split(predictors, labels):
        print('Train: ' + str(train_data))
        print('Test: ' + str(test_data))
        for model in models:
            # Fit
            model.fit(predictors[train_data], labels[train_data], epochs=n_epochs, batch_size=batch_sz, verbose=0)
            # Evaluate
            scores = model.evaluate(predictors[test_data], labels[test_data], batch_size=batch_sz, verbose=0)
            accuracies.append(scores)  # TODO: Set to structured numpy array, fieldnames as model names
    print('Avg acc: ' + str(np.mean(accuracies)))
    return accuracies


def k_fold(predictors, labels, n_splits):
    '''
    Splits predictors and labels into a number of testing groups
    :param predictors: All of the predictors data to be spllit
    :param labels: All of the label data to be split
    :param n_splits: 
    :return: AEach fold is a nested tuple, of (train_data, test_data) where
    train_data = (train_predictors, train_labels) and test_data = (test_predictors, test_labels)
    '''
    folds = []
    for i in range(0, n_splits):
        train_data = (
            predictors[np.mod([i for i in range(0, len(labels))], n_splits) != 0],
            labels[np.mod([i for i in range(0, len(labels))], n_splits) != 0]
        )
        test_data = (
            predictors[i::n_splits],
            labels[i::n_splits]
        )
        folds.append((train_data, test_data))
    return folds

# TODO: mean prediction and LDA (and Naive Bayes?)

print('Imported cnv_eval')
