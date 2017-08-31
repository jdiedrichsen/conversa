''' cnv_model - Manages Conversa models '''

from abc import ABCMeta, abstractmethod  # Abstract base class import
import numpy as np
import pandas as pd

# TODO: Add functions to write to file
# TODO: Naive Bayes classifier, LDA

# Models must implement subsequence division on their own, use cnv_data.to_seqs if required


class Model(metaclass=ABCMeta):
    '''
    Abstract class for all nn_models to implement
    Expected methods to implement:
        learn(x, y)
        predict(x)
    '''
    @abstractmethod
    def __init__(self):
        '''
        Initialize the model
        '''

    @abstractmethod
    def learn(self, predictors, labels):
        '''
        Fit/train/learn the corresponding predictors and labels
        :param predictors: Inputs to the model (x values) in a pandas DataFrame
        :param labels: True outputs corresponding to the predictors (y values) in a pandas DataFrame
        '''

    @abstractmethod
    def predict(self, predictors):
        '''
        Predict labels corresponding to the predictors
        :param predictors: Inputs to the model (x values) in a pandas DataFrame
        :return: Estimated labels (y values) in a pandas DataFrame
        '''


# TODO - Reimplement for DataFrames
# TODO: Fix bug with occasional off-by-one prediction
class ZeroModel:
    '''
    Always predicts zero
    '''

    def __init__(self):
        Model.__init__(self)
        self._y_shape = tuple()

    def learn(self, predictors, labels):
        from copy import deepcopy
        self._y_shape = deepcopy(labels.shape)

    def predict(self, x):
        return np.zeros(self._y_shape)


# TODO - Reimplement for DataFrames
class MeanModel(Model):
    '''
    Always predicts the mean of values it's been trained on
    '''

    def __init__(self):
        Model.__init__(self)
        self.mean = None

    def learn(self, predictors, labels):
        from numpy import mean
        self.mean = mean(labels)

    def predict(self, predictors):
        return self.mean


class LinearModel:
    pass


# TODO - Major refactoring
class SVMModel:

    def __init__(self):
        Model.__init__(self)
        from sklearn.svm import SVC
        self.mdl = SVC()
        self.label_columns = None

    def learn(self, predictors, labels):
        self.mdl.fit(predictors.values, np.ravel(labels.values))
        self.label_columns = labels.columns.tolist()

    def predict(self, predictors):
        return pd.DataFrame(self.mdl.predict(predictors.values), columns=self.label_columns)


class NeuralNetworkModel:
    pass

