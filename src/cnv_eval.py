''' cnv_eval - Model evaluation tools for Conversa '''
import numpy as np
import pandas as pd
from copy import deepcopy

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

# Header strings
PID_H_STR = 'pid'
CAM_H_STR = 'cam'
BHV_H_STR = 'behaviour'
MDL_H_STR = 'model'
FLD_H_STR = 'fold_no'
ACC_H_STR = 'accuracy'
# LOSS_STR = 'loss'


# Only works when elements in prediction and actual are in range [0, 1]
# TODO: Documentation
def accuracy(prediction, actual):
    if not prediction.shape == actual.shape:
        raise RuntimeError('Comparing prediction and actual value of different shape')
    return np.mean(1 - np.absolute(np.round(prediction) - actual))


# def rmse(prediction, actual):
#     return np.sqrt(np.mean(np.square(prediction - actual)))


# TODO: Documentation
def evaluate(model, predictors, labels, eval_func=accuracy):
    # try:
    #     from cnv_data import destructure
    # except ImportError:
    #     print('Unable to import cnv_data.load_subject')
    # print(type(predicted_labels))
    # print(destructure(predicted_labels).shape)
    # print(labels.shape)
    # print(destructure(labels).shape)
    predicted_labels = model.predict(predictors)
    return eval_func(predicted_labels, labels)


# In each epoch of training, all data is used
# The batch size specifies how many data points are given to the model at once
# E.g. with 135 data points, n_epochs = 10, and batch_sz = 20, you get 7 batches per epoch where 6 of the batches are of
# size 20 and one is of size 15 (so every data point is used once). The model trains on this data 10 times (10 epochs)
# and each time it divides the data into 7 batches

def eval_models(models,
                predictors,
                labels,
                n_folds=5,
                # train_n_epochs=10,  # Original setting
                # train_batch_sz=10,
                train_n_epochs=1,
                train_batch_sz=1,
                return_data_frame=True,  # TODO: Add to doc
                verbose=0):
    '''
    Evaluates models given predictor and label data to train and test the models on
    :param return_data_frame: 
    :param models: The models to evaluate
    :param predictors: Predictors to test the models on
    :param labels: Labels to test the models on
    :param n_folds: The number of folds to test the data on, defaults to 5
    :param train_n_epochs: The number of passes each models gets on the data, defaults to 10
    :param train_batch_sz: The number of data points to train each model on at once, defaults to 10
    :param verbose: The verbosity level of model training and testing - note that model console output often conflicts
    with outputs from cnv_eval - defaults to 0 (not verbose)
    :return: A pandas DataFrame with columns fold_no, model_no, and accuracy
    '''
    # TODO: Update doc
    # TODO: Add verbose flags, vprint function, replace prints with vprint

    folds = k_fold(predictors, labels, n_folds)
    eval_results = dict([
        (FLD_H_STR, []),
        (MDL_H_STR, []),
        (ACC_H_STR, [])
        # (LOSS_STR, [])
    ])
    for model_no in range(0, len(models)):

        # Select model
        print('Model: ' + str(model_no+1) + '/' + str(len(models)))
        # model = deepcopy(models[model_no])  # Model resets every fold - TODO: Ask what behaviour should be
        model = models[model_no]

        for fold_no in range(0, len(folds)):

            # Select fold
            print('\tFold: ' + str(fold_no+1) + '/' + str(len(folds)))
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
            # (_, acc) = model.evaluate(test_predictors, test_labels, batch_size=test_n_batch_sz, verbose=verbose)
            acc = accuracy(prediction=model.predict(test_predictors), actual=test_labels)
            print('\t\t\tAccuracy: ' + str(acc))

            # Set accuracy
            eval_results[MDL_H_STR].append(model_no + 1)
            eval_results[FLD_H_STR].append(fold_no + 1)
            eval_results[ACC_H_STR].append(acc)
            # evaluation[LOSS_STR].append(loss)

    if return_data_frame:
        output = order_by_fields(pd.DataFrame(eval_results), [MDL_H_STR])
    else:
        output = eval_results
    print('Evaluation complete')
    # print(eval_results)  # For debugging
    return output


# TODO: Documentation
def order_by_fields(data, field_names):
    '''
    Re-orders the columns of data according to field_names
    Refactored from https://stackoverflow.com/a/25023460/7195043
    '''
    back_fields = [col for col in data.columns if col not in field_names]
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
            # predictors[np.mod([units_exp for units_exp in range(0, len(labels))], n_folds) != 0],
            # labels[np.mod([units_exp for units_exp in range(0, len(labels))], n_folds) != 0]
        )
        folds.append((train_data, test_data))
    return folds


# Subjects are tuples of (pid, cam), where pid and cam are numbers, like (2024, 2)
# Set behavs to None for all behavs being trained on, otherwise provide iterable of strings
# TODO: Documentation
def eval_models_on_subjects(models, subjects, behaviours=None, timesteps=30, n_folds=5, verbose=0):

    eval_results = dict([
        (PID_H_STR, []),
        (CAM_H_STR, []),
        (BHV_H_STR, []),
        (MDL_H_STR, []),
        (FLD_H_STR, []),
        (ACC_H_STR, []),
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
            behaviours = labels.dtype.names

        for behav_name in behaviours:

            print('Behaviour: ' + str(behav_name))

            predict_seqs = to_subseqs(predicts, timesteps)
            label_seqs = to_subseqs(add_dim(labels[behav_name]), timesteps)

            # # Old implementation kept using same models for all subjects and behaviours
            # sub_eval_results = eval_models(models, predict_seqs, label_seqs, return_data_frame=False)
            # Evaluate copy of models on the subject and behaviour - different instance of model used for each subject
            # and behaviour
            # TODO: Determine best behaviour and ask about preferred implementation
            # specialist_models = deepcopy(models)  # These models specialize for each subject and behaviour
            sub_eval_results = eval_models(models, predict_seqs, label_seqs,
                                           return_data_frame=False,
                                           n_folds=n_folds,
                                           verbose=verbose)

            # Add results to over evaluation results
            n_rows = len(sub_eval_results[ACC_H_STR])
            eval_results[PID_H_STR].extend([pid] * n_rows)
            eval_results[CAM_H_STR].extend([cam] * n_rows)
            eval_results[BHV_H_STR].extend([behav_name] * n_rows)
            eval_results[MDL_H_STR].extend(sub_eval_results[MDL_H_STR])
            eval_results[FLD_H_STR].extend(sub_eval_results[FLD_H_STR])
            eval_results[ACC_H_STR].extend(sub_eval_results[ACC_H_STR])

    eval_df = order_by_fields(pd.DataFrame(eval_results), [PID_H_STR, CAM_H_STR, BHV_H_STR, MDL_H_STR, FLD_H_STR, ACC_H_STR])
    eval_df.sort_values([BHV_H_STR, MDL_H_STR])

    print('Models evaluated on subjects')
    return eval_df


# Takes the average on some fields in a dataframe and returns a different dataframe
# E.g.
# df = [('a', [1, 1, 2, 2]), ('b', [7, 1 , 0, 10])], average_fields = ['b']
# returns [('a', [1, 2]), ('b', [4, 5])]
# TODO
def average_on(df, average_fields):
    pass


# TODO: Implementation and documentation
def summary(eval_results, average_on=[FLD_H_STR]):
    '''
    Returns a summarized version of model evaluations
    :param eval_results: 
    :return: 
    '''

    summary_dict = dict([

    ])

    pd.DataFrame(summary_dict)

    eval_results = pd.DataFrame()

    min_model_no = eval_results[MDL_H_STR].min()
    max_model_no = eval_results[MDL_H_STR].max()
    for i in range(min_model_no, max_model_no):
        eval_results.loc[
            eval_results[MDL_H_STR] == i
            ].mean()

    pass


# TODO: Update doc.md


print('Imported cnv_eval')
