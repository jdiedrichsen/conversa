import numpy as np

# TODO: Add verbose flags and vprint function
# TODO: mean prediction and LDA (and Naive Bayes?)


def eval_models(models, predictors, labels, n_folds=5, train_n_epochs=10, train_batch_sz=10, test_n_epochs=10,  # TODO
                test_n_batch_sz=10):
    '''
    Evaluates models given predictor and label data to train and test the models on
    :param models: 
    :param predictors: 
    :param labels: 
    :param n_folds: 
    :param train_n_epochs: 
    :param train_batch_sz: 
    :param test_n_epochs: 
    :param test_n_batch_sz: 
    :return: 
    '''
    folds = k_fold(predictors, labels, n_folds)
    accuracies = []
    for model in models:
        print('\n\nMoving to next model')
        for (train_data, test_data) in folds:
            print('\nMoving to next fold')
            (train_predictors, train_labels) = train_data
            (test_predictors, test_labels) = test_data
            # Train
            model.fit(train_predictors, train_labels, epochs=train_n_epochs, batch_size=train_batch_sz, verbose=0)
            # Test
            scores = model.evaluate(test_predictors, test_labels, batch_size=test_n_batch_sz, verbose=0)
            accuracies.append(scores)  # TODO: Set to structured numpy array, fieldnames as spec_model names
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
