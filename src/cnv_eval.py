''' cnv_eval - Model evaluation tools for Conversa '''
import numpy as np
import pandas as pd
from copy import copy

__author__ = 'Shayaan Syed Ali'
# __copyright__ = ''
__credits__ = ['Shayaan Syed Ali']
__maintainer__ = 'Shayaan Syed Ali'
__email__ = 'shayaan.syed.ali@gmail.com'
__status__ = 'Development'
# __license__ = ''
# __version__ = ''

# TODO: Mean prediction and LDA (and Naive Bayes?)
# TODO: In doc add function guide/map

# Constants
PID_STR = 'pid'
CAM_STR = 'cam'
BEHAV_STR = 'behaviour'
MODEL_NO_STR = 'model_no'
FOLD_NO_STR = 'fold_no'
ACC_STR = 'accuracy'
# LOSS_STR = 'loss'


# TODO: Fix evaluation function to use model.predict and error_fun, also make it work with structured arrays

# # TODO
# def diff(x, y):
#     '''
#     Returns the difference between two structured numpy arrays with the same fields
#     '''
#     pass
#
#
# def pct_diff(prediction, actual):
#     return np.mean(prediction - actual)
#
#
# def rmse(prediction, actual):
#     return np.sqrt(np.mean(np.square(prediction - actual)))
#
#
# def evaluate(model, predictors, labels, error_func=pct_diff):
#     # TODO: Add check if structured or not
#     # try:
#     #     from cnv_data import destructure
#     # except ImportError:
#     #     print('Unable to import cnv_data.load_subject')
#     # print(type(predicted_labels))
#     # print(destructure(predicted_labels).shape)
#     # print(labels.shape)
#     # print(destructure(labels).shape)
#     predicted_labels = model.predict(predictors)
#     return 1 - error_func(predicted_labels, labels)


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
                return_data_frame=True,
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
    # TODO: Update doc
    # TODO: Add verbose flags, vprint function, replace prints with vprint

    folds = k_fold(predictors, labels, n_folds)
    eval_results = dict([
        (FOLD_NO_STR, []),
        (MODEL_NO_STR, []),
        (ACC_STR, [])
        # (LOSS_STR, [])
    ])
    for model_no in range(0, len(models)):
        print('Moving to model: ' + str(model_no+1) + '/' + str(len(models)))
        for fold_no in range(0, len(folds)):
            print('\tMoving to fold: ' + str(fold_no+1) + '/' + str(len(folds)))
            # model = copy(models[model_no])  # Model resets every fold - TODO: Ask what behaviour should be
            model = models[model_no]
            fold = folds[fold_no]
            # Unpack data from fold
            (train_data, test_data) = fold
            (train_predictors, train_labels) = train_data
            (test_predictors, test_labels) = test_data
            # Train
            print('\t\tTraining')
            model.fit(train_predictors, train_labels, epochs=train_n_epochs, batch_size=train_batch_sz, verbose=verbose)
            # Test
            print('\t\tEvaluating')
            (_, accuracy) = model.evaluate(test_predictors, test_labels, batch_size=test_n_batch_sz, verbose=verbose)
            # accuracy = evaluate(model, test_predictors, test_labels)
            # Set accuracy
            eval_results[MODEL_NO_STR].append(model_no+1)
            eval_results[FOLD_NO_STR].append(fold_no+1)
            eval_results[ACC_STR].append(accuracy)
            # evaluation[LOSS_STR].append(loss)
    if return_data_frame:
        output = order(pd.DataFrame(eval_results), [MODEL_NO_STR, FOLD_NO_STR, ACC_STR])
    else:
        output = eval_results
    print('Evaluation complete')
    return output


def order(data, field_names):
    '''
    Re-orders the columns of data according to field_names
    Refactored from https://stackoverflow.com/a/25023460/7195043
    '''
    back_fields =[col for col in data.columns if col not in field_names]
    data = data[field_names + back_fields]
    return data


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


# Subjects are tuples of (pid, cam), where pid and cam are numbers, like (2024, 2)
# Set behavs to None for all behavs being trained on, otherwise provide iterable of strings
def eval_models_on_subjects(models, subjects, behaviours=None, timesteps=30):

    eval_results = dict([
        (PID_STR, []),
        (CAM_STR, []),
        (BEHAV_STR, []),
        (MODEL_NO_STR, []),
        (FOLD_NO_STR, []),
        (ACC_STR, []),
    ])

    try:
        from cnv_data import load_subject, add_dim, to_subseqs, destructure
    except ImportError:
        print('Unable to import cnv_data functions')

    for (pid, cam) in subjects:
        print('Subject: pid' + str(pid) + 'cam' + str(cam))
        (predicts, labels) = load_subject(pid, cam)

        # Set behavs if not provided
        if behaviours is None:
            behaviours = labels.dtype.names

        for behav_name in behaviours:
            print('Behaviour: ' + str(behav_name))
            predict_seqs = to_subseqs(predicts, timesteps)
            label_seqs = to_subseqs(add_dim(labels[behav_name]), timesteps)
            # # Old implementation kept using same models for all subjects and behaviours
            # sub_eval_results = eval_models(models, predict_seqs, label_seqs, return_data_frame=False)
            # Evaluate copy of models on the subject and behaviour - different instance of model used for each subject
            # and behaviour
            sub_eval_results = eval_models(copy(models), predict_seqs, label_seqs, return_data_frame=False)
            # Add results to over evaluation results
            n_rows = len(sub_eval_results[ACC_STR])
            eval_results[PID_STR].extend([pid]*n_rows)
            eval_results[CAM_STR].extend([cam]*n_rows)
            eval_results[BEHAV_STR].extend([behav_name]*n_rows)
            eval_results[MODEL_NO_STR].extend(sub_eval_results[MODEL_NO_STR])
            eval_results[FOLD_NO_STR].extend(sub_eval_results[FOLD_NO_STR])
            eval_results[ACC_STR].extend(sub_eval_results[ACC_STR])

    print('Models evaluated on subjects')
    return order(pd.DataFrame(eval_results), [PID_STR, CAM_STR, BEHAV_STR, MODEL_NO_STR, FOLD_NO_STR, ACC_STR])


print('Imported cnv_eval')
