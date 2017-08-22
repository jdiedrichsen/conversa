''' cnv_model - Manages Conversa models '''

from abc import ABCMeta, abstractmethod  # Abstract base class import


class Model(metaclass=ABCMeta):
    '''
    Abstract class for all models to implement
    Expected methods to implement:
        learn(x, y)
        predict(x)
    '''
    @abstractmethod
    def __init__(self):
        '''
        
        '''

    @abstractmethod
    def learn(self, x, y):
        '''
        
        :param x: 
        :param y: 
        :return: 
        '''

    @abstractmethod
    def predict(self, x):
        '''
        
        :param x: 
        :return: 
        '''


class NullModel:
    pass


class MeanModel:
    pass


class LinearModel:
    pass


class SVMModel:
    pass


class NeuralNetworkModel:
    pass

