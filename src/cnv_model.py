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

    @abstractmethod
    def summary(self):
        '''
        A summary of the model
        :return: A string which summarizes the model and parameters
        '''

    def name(self):
        '''
        Returns the name of the model, should fit on one line
        :return: The name of the model as a short string
        '''
        return self.__class__.__name__


# TODO - Reimplement for DataFrames
class MeanModel(Model):
    '''
    Always predicts the mean of values it's been trained on
    '''

    def __init__(self):
        Model.__init__(self)
        self._mean = None
        self._label_columns = None

    def learn(self, predictors, labels):
        from numpy import mean
        self._mean = mean(labels)
        self._label_columns = labels.columns.tolist()

    def predict(self, predictors):
        return pd.DataFrame(np.full((predictors.shape[0], len(self._label_columns)), self._mean), columns=self._label_columns)

    def summary(self):
        return 'MeanModel\n\tmean=' + str(self._mean)


class SVMModel:

    def __init__(self):
        Model.__init__(self)
        from sklearn.svm import SVC
        self._mdl = SVC()
        self.__single_label = None
        self._label_columns = None

    def learn(self, predictors, labels):
        # The SVC (and other models in sklearn) require that all data used with the fit function have multiple class
        # labels, so we must ensure this is the case if we pass the data to the fit function, otherwise we will just
        # predict the only label present, __single_label
        self.__single_label = None if labels.nunique()[0] > 1 else labels[labels.columns.tolist()[0]].iloc[0]
        if self.__single_label is None:  # Fit only if there are multiple labels in the data
            self._mdl.fit(predictors.values, np.ravel(labels.values))
        self._label_columns = labels.columns.tolist()

    def predict(self, predictors):
        if self.__single_label is None:
            return pd.DataFrame(self._mdl.predict(predictors.values), columns=self._label_columns)
        else:  # Predict the single label from training data
            return pd.DataFrame(np.full((predictors.shape[0], len(self._label_columns)), self.__single_label), columns=self._label_columns)

class NeuralNetworkModel:
    pass

