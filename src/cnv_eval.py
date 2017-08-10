from sklearn.model_selection import StratifiedKFold
import numpy


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
    print('Avg acc: ' + str(numpy.mean(accuracies)))
    return accuracies




# def eval_2(models, predictors, labels)
#
#
#
# def k_fold(n_splits, max_index):
#     folds =
#     for i in range()

# TODO: mean prediction and LDA
