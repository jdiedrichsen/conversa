''' cnv_eval - Model evaluation tools for Conversa '''

__author__ = 'Shayaan Syed Ali'
# __copyright__ = ''
__credits__ = ['Shayaan Syed Ali']
__maintainer__ = 'Shayaan Syed Ali'
__email__ = 'shayaan.syed.ali@gmail.com'
__status__ = 'Development'
# __license__ = ''
# __version__ = ''

import numpy as np
import pandas as pd

# TODO: Add verbose flags and vprint function
# TODO: mean prediction and LDA (and Naive Bayes?)
# TODO: In doc add funciton guide/map


# In each epoch of training, all data is used
# The batch size specifies how many data points are given to the model at once
# E.g. with 135 data points, n_epochs = 10, and batch_sz = 20, you get 7 batches per epoch where 6 of the batches are of
# size 20 and one is of size 15 (so every data point is used once). The model trains on this data 10 times (10 epochs)
# and each time it divides the data into 7 batches

def eval_models(models,
                predictors,
                labels,
                n_folds=5,
                train_n_epochs=10,
                train_batch_sz=10,
                test_n_batch_sz=1,
                verbose=0):
    '''
    Evaluates models given predictor and label data to train and test the models on
    :param models: The models to evaluate
    :param predictors: Predictors to test the models on
    :param labels: Labels to test the models on
    :param n_folds: The number of folds to test the data on, defaults to 5
    :param train_n_epochs: The number of passes each models gets on the data, defaults to 10
    :param train_batch_sz: The number of data points to train each model on at once, defaults to 10
    :param test_n_batch_sz: The number of data points to test each model on at once, defaults to 1
    :param verbose: The verbosity level of training and testing - note that model console output often conflicts with 
    outputs from cnv_eval - defaults to 0 (not verbose)
    :return: 
    '''
    # TODO: Change return to pandas DataFrame
    folds = k_fold(predictors, labels, n_folds)
    pd.DataFrame()
    accuracies = []
    for model in models:
        print('\nMoving to next model')
        for (train_data, test_data) in folds:
            print('\nMoving to next fold')
            (train_predictors, train_labels) = train_data
            (test_predictors, test_labels) = test_data
            # Train
            model.fit(train_predictors, train_labels, epochs=train_n_epochs, batch_size=train_batch_sz, verbose=verbose)
            # Test
            (loss, accuracy) = model.evaluate(test_predictors, test_labels, batch_size=test_n_batch_sz, verbose=verbose)
            accuracies.append(accuracy)
    return accuracies


def k_fold(predictors, labels, n_folds):
    '''
    Splits predictors and labels into a number of testing groups
    :param predictors: All of the predictors data to be split
    :param labels: All of the label data to be split
    :param n_folds: The number of folds to split the data into
    :return: Each fold is a nested tuple, of (train_data, test_data) where
    train_data = (train_predictors, train_labels) and test_data = (test_predictors, test_labels)
    '''
    # TODO: Change to pandas DataFrame
    folds = list()
    for i in range(0, n_folds):
        test_data = (
            predictors[i::n_folds],
            labels[i::n_folds]
        )
        # Here pred is short for predictor and labl is short for label
        # Used to avoid confusion with predictors and labels
        train_data = (
            np.array([pred for (j, pred) in enumerate(predictors) if (j-i) % n_folds != 0]),
            np.array([labl for (j, labl) in enumerate(labels) if (j-i) % n_folds != 0])
            # predictors[np.mod([i for i in range(0, len(labels))], n_folds) != 0],
            # labels[np.mod([i for i in range(0, len(labels))], n_folds) != 0]
        )
        folds.append((train_data, test_data))
    return folds


def print_summary(scores):
    pass


print('Imported cnv_eval')
