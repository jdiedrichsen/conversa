''' cnv_model - Manages Conversa models '''

from abc import ABCMeta, abstractmethod  # Abstract base class import
import numpy as np


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
        Initialize the model, abc Model requires no instantiations
        '''

    @abstractmethod
    def learn(self, predictors, labels):
        '''
        Fit/train/learn the corresponding predictors and labels
        :param predictors: Inputs to the model (x values)
        :param labels: True outputs corresponding to the predictors (y values)
        '''

    @abstractmethod
    def predict(self, predictors):
        '''
        Predict labels corresponding to the predictors
        :param predictors: Inputs to the model (x values)
        :return: Estimated labels (y values)
        '''


class NullModel:

    def __init__(self):
        Model.__init__(self)
        self._y_shape = tuple()

    def learn(self, predictors, labels):
        from copy import deepcopy
        self._y_shape = deepcopy(labels.shape)

    def predict(self, x):
        return np.zeros(self._y_shape)


# TODO - dbug
class MeanModel(Model):

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
        # setattr(self, 'mdl', SVC())

    def learn(self, predictors, labels):
        try:
            from cnv_data import destructure
        except ImportError:
            print('Unable to import cnv_data')
        p_2 = destructure(predictors)
        p_2_s = p_2.shape
        p_2 = np.reshape(p_2, (p_2_s[0], p_2_s[2]))
        # print(predictors.shape)
        # print(p_2.shape)
        l_2 = labels
        l_2_s = l_2.shape
        l_2 = np.reshape(l_2, (l_2_s[0], l_2_s[2]))
        # print(labels.shape)
        # print(l_2.shape)
        self.mdl.fit(p_2, l_2)

    def predict(self, predictors):
        try:
            from cnv_data import destructure
        except ImportError:
            print('Unable to import cnv_data')
        p_2 = destructure(predictors)
        p_2_s = p_2.shape
        p_2 = np.reshape(p_2, (p_2_s[0], p_2_s[2]))
        return np.reshape(self.mdl.predict(p_2), (len(self.mdl.predict(p_2)), 1, 1))  # TODO - Make flexible


class NeuralNetworkModel:
    pass

