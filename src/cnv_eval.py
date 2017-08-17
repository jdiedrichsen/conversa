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

# TODO: Mean prediction and LDA (and Naive Bayes?)
# TODO: In doc add function guide/map

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
    :param verbose: The verbosity level of model training and testing - note that model console output often conflicts
    with outputs from cnv_eval - defaults to 0 (not verbose)
    :return: A pandas DataFrame with columns fold_no, model_no, and accuracy
    '''
    # TODO: Add verbose flags and vprint function

    # Function constants
    FOLD_NO_STR = 'fold_no'
    MODEL_NO_STR = 'model_no'
    ACC_STR = 'accuracy'
    LOSS_STR = 'loss'

    folds = k_fold(predictors, labels, n_folds)
    evaluation = dict([
        (FOLD_NO_STR, []),
        (MODEL_NO_STR, []),
        (ACC_STR, []),
        (LOSS_STR, [])
    ])
    for model_no in range(0, len(models)):
        print('\nMoving to model:\t' + str(model_no+1))
        model = models[model_no]
        for fold_no in range(0, len(folds)):
            print('\nMoving to next fold:\t' + str(fold_no+1))
            fold = folds[fold_no]
            # Unpack data from fold
            (train_data, test_data) = fold
            (train_predictors, train_labels) = train_data
            (test_predictors, test_labels) = test_data
            # Train
            model.fit(train_predictors, train_labels, epochs=train_n_epochs, batch_size=train_batch_sz, verbose=verbose)
            # Test
            (loss, accuracy) = model.evaluate(test_predictors, test_labels, batch_size=test_n_batch_sz, verbose=verbose)
            # Set accuracy and loss
            evaluation[FOLD_NO_STR].append(fold_no)
            evaluation[MODEL_NO_STR].append(model_no)
            evaluation[ACC_STR].append(accuracy)
            evaluation[LOSS_STR].append(loss)
    return pd.DataFrame(data=evaluation)


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
