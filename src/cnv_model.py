''' cnv_model - Manages Conversa models '''


class Model:
    '''
    Abstract class for all models to implement
    Expected methods to implement:
        learn(x, y)
        predict(x)
    '''
    def learn(self, x, y):
        raise('learn method not implemented!')

    def predict(self, x):
        raise('learn method not implemented!')


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

