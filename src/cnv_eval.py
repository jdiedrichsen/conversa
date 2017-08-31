''' cnv_eval - Model evaluation tools for Conversa '''

import numpy as np
import pandas as pd
from copy import copy, deepcopy
from sklearn.model_selection import StratifiedKFold

__author__ = 'Shayaan Syed Ali'
# __copyright__ = ''
__credits__ = ['Shayaan Syed Ali']
__maintainer__ = 'Shayaan Syed Ali'
__email__ = 'shayaan.syed.ali@gmail.com'
__status__ = 'Development'
# __license__ = ''
# __version__ = ''

# TODO: In doc add function guide/map
# TODO: Add vprint and verbose flags

# Constants

# Header strings
_PID_H_STR = 'pid'
_CAM_H_STR = 'cam'
_BEHAV_H_STR = 'behaviour'
_MODEL_H_STR = 'model'
_FOLD_H_STR = 'fold_no'
_ACCURACY_H_STR = 'accuracy'

HEADER_STRINGS = [  # TODO: Convert to dict and use as such in code
    _PID_H_STR,
    _CAM_H_STR,
    _BEHAV_H_STR,
    _MODEL_H_STR,
    _FOLD_H_STR,
    _ACCURACY_H_STR,
]


# TODO: Make capable of dealing with larger range of value (more than [0, 1]
def accuracy(predicted, true, rounding=True):
    '''
    Determines the accuracy of a predicted value against an actual value for values in the range [0, 1]
    Requires that the predicted and true values are numpy arrays (or of classes that work with numpy functions) and that
    they are of the same shape
    :param predicted: The predicted value(s) as a numpy array, same shape as true
    :param true: The actual value(s) as a numpy array, same shape as predicted
    :param rounding: Whether to round predicted values or not, defaults to True
    :return: The accuracy of the prediction against the true value, specifically the 
    '''
    if not predicted.shape == true.shape:
        raise RuntimeError('Prediction shape is ' + str(predicted.shape) + ' while true has shape ' + str(true.shape))

    if rounding:
        abs_err = np.absolute(np.round(predicted.values) - true.values)
    else:
        abs_err = np.absolute(predicted.values - true.values)

    return 1 - np.mean(abs_err)


def eval_models(models, predictors, labels, n_folds=5, return_data_frame=True, verbose=0):  # TODO: Implement verbose
    '''
    Evaluates nn_models given predictor and label data to train and test the nn_models on
    :param models: The nn_models to evaluate
    :param predictors: Predictors to test the nn_models on
    :param labels: Labels to test the nn_models on
    :param n_folds: The number of folds to test the data on, defaults to 5
    :param return_data_frame: Whether to return the evaluation data in a pandas DataFrame or a Python dict
    :param verbose: The verbosity level of model training and testing - note that model console output often conflicts
    with outputs from cnv_eval - defaults to 0 (not verbose)
    :return: A pandas DataFrame with columns fold_no, model_no, and accuracy or a dict if return_data_frame=False
    '''

    # Set up eval_results as dict and convert to pandas DataFrame if return_data_frame is True
    # It is significantly faster to work this way
    eval_results = dict([
        (_FOLD_H_STR, []),
        (_MODEL_H_STR, []),
        (_ACCURACY_H_STR, [])
    ])

    for model_no in range(0, len(models)):

        # Select model
        model = models[model_no]
        print('Model: ' + str(model_no+1) + '/' + str(len(models)) + ', ' + str(model))
        # model = deepcopy(nn_models[model_no])  # Resets model on each fold
        # TODO: Determine applicable behaviour or parameterize

        k_fold = StratifiedKFold(n_splits=n_folds)

        fold_no = 0

        for train_index, test_index in k_fold.split(
                np.ravel(np.zeros(
                    (predictors.shape[0], 1)
                )),
                np.ravel(np.zeros(
                    (labels.shape[0], 1)
                ))):

            # Select fold
            print('\tFold: ' + str(fold_no+1) + '/' + str(n_folds), end='', flush=True)

            # Unpack data from fold
            train_predictors = predictors.iloc[train_index]
            train_labels = labels.iloc[train_index]
            test_predictors = predictors.iloc[test_index]
            test_labels = labels.iloc[test_index]

            # Train
            print(', training', end='', flush=True)
            # print(train_predictors)
            model.learn(train_predictors, train_labels)

            # Test
            print(', evaluating', end='', flush=True)
            acc = accuracy(predicted=model.predict(test_predictors), true=test_labels)
            print(', accuracy: ' + str(acc), flush=True)

            # Set accuracy
            eval_results[_MODEL_H_STR].append(model_no + 1)
            eval_results[_FOLD_H_STR].append(fold_no + 1)
            eval_results[_ACCURACY_H_STR].append(acc)

            fold_no = fold_no + 1

    # Return applicable DataFrame or dict
    return eval_results if return_data_frame else order_fields(pd.DataFrame(eval_results).sort_values(_MODEL_H_STR), [_MODEL_H_STR])


def order_fields(df, priority_fields):
    '''
    Re-orders the columns of a pandas DataFrame according to column_names
    Refactored from https://stackoverflow.com/a/25023460/7195043
    :param df: The DataFrame whose columns are to be reordered
    :param priority_fields: The fields to bring to the left in order, does not need to include all columns - others will
    be added at the back
    :return: The DataFrame with reordered columns
    '''
    remaining_fields = [col for col in df.columns if col not in priority_fields]
    df = df[priority_fields + remaining_fields]
    return df

# # DEPRECATED - Removed in favour of sklearn's StratifiedKFold
# def k_folds(predictors, labels, n_folds):
#     '''
#     Splits predictors and labels into a number of testing groups
#     :param predictors: All of the predictors data to be split
#     :param labels: All of the label data to be split
#     :param n_folds: The number of folds to split the data into
#     :return: Each fold is a nested tuple, of (train_data, test_data) where
#     train_data = (train_predictors, train_labels) and test_data = (test_predictors, test_labels)
#     '''
#
#     pass
#
#     folds = list()
#     for i in range(0, n_folds):
#
#         # Test data
#         test_predictors = None
#         test_labels = None
#         test_data = (test_predictors, test_labels)
#
#         # Train data
#         train_predictors = None
#         train_labels = None
#         train_data = (train_predictors, train_labels)
#
#         folds.append((train_data, test_data))
#
#     return folds


def eval_models_on_subjects(models, subjects, behaviours=None, n_folds=5, verbose=0):
    '''
    Runs evaluation for a list of models on a list of subjects
    :param models: Model objects, should implement Model abstract base class from cnv_model
    :param subjects: A tuple of the form (pid, cam), where pid and cam denote the pid number and cameras number
    respectively, like (2024, 2)
    :param behaviours: Behaviours to train on, leave as None for training on all behaviour separately
    :param n_folds: The number of folds for the k-folds cross validation algorithm
    :param verbose: How much debugging information is given, higher numbers giv more info, zero is the minimum and gives
    only errors
    :return: A pandas DataFrame summarizing all the results
    '''

    eval_results = dict([
        (_PID_H_STR, []),
        (_CAM_H_STR, []),
        (_BEHAV_H_STR, []),
        (_MODEL_H_STR, []),
        (_FOLD_H_STR, []),
        (_ACCURACY_H_STR, []),
    ])

    try:
        from cnv_data import load_subject, add_dim, to_subseqs
    except ImportError:
        print('Unable to import cnv_data functions')

    for (pid, cam) in subjects:
        print('Subject: pid' + str(pid) + 'cam' + str(cam))
        (predicts, labels) = load_subject(pid, cam)

        # Set behavs if not provided
        if behaviours is None:
            behaviours = labels.columns.tolist()

        for behav_name in behaviours:

            print('Behaviour: ' + str(behav_name))

            behav_labels = pd.DataFrame(labels[behav_name], columns=[behav_name])  # DataFrame of single behaviour
            sub_eval_results = eval_models(models, predicts, behav_labels, return_data_frame=False, n_folds=n_folds, verbose=verbose)

            # Add results to over evaluation results
            n_rows = len(sub_eval_results[_ACCURACY_H_STR])
            eval_results[_PID_H_STR].extend([pid] * n_rows)
            eval_results[_CAM_H_STR].extend([cam] * n_rows)
            eval_results[_BEHAV_H_STR].extend([behav_name] * n_rows)
            eval_results[_MODEL_H_STR].extend(sub_eval_results[_MODEL_H_STR])
            eval_results[_FOLD_H_STR].extend(sub_eval_results[_FOLD_H_STR])
            eval_results[_ACCURACY_H_STR].extend(sub_eval_results[_ACCURACY_H_STR])

    eval_df = order_fields(pd.DataFrame(eval_results), [_PID_H_STR, _CAM_H_STR, _BEHAV_H_STR, _MODEL_H_STR, _FOLD_H_STR, _ACCURACY_H_STR])
    eval_df.sort_values([_MODEL_H_STR, _BEHAV_H_STR])

    print('Models evaluated on subjects')
    return eval_df


def summary(eval_results, drop_fields=[_FOLD_H_STR]):
    '''
    Returns a summarized version of model evaluations which averages the accuracy of models across folds
    :param eval_results: The DataFrame to summarize
    :return: A summary DataFrame
    '''
    for drop_field in drop_fields:
        summary_df = eval_results.drop(drop_field, 1)
    summary_df = (summary_df.groupby([_PID_H_STR, _CAM_H_STR, _BEHAV_H_STR, _MODEL_H_STR]).mean())
    return summary_df


print('Imported cnv_eval')
