''' cnv_net - Provides neural net utilities for Conversa '''

import math, random
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, GRU, LSTM
from keras.layers.convolutional import Conv1D, Conv2D

__author__ = 'Shayaan Syed Ali'
# __copyright__ = ''
__credits__ = ['Shayaan Syed Ali']
__maintainer__ = 'Shayaan Syed Ali'
__email__ = 'shayaan.syed.ali@gmail.com'
__status__ = 'Development'
# __license__ = ''
# __version__ = ''

# Constants

UNITS_MAX = 32
UNITS_N_SAMPLES = 8
DROPOUTS_N_SAMPLES = 4

# General available values for layer parameters
units = [2**i for i in range(0, int(math.log(UNITS_MAX, 2))+1)]
# units = np.linspace(1, UNITS_MAX, num=DROPOUTS_N_SAMPLES)
activations = ['\'relu\'', '\'tanh\'', '\'sigmoid\'', '\'hard_sigmoid\'', '\'softmax\'']
# activations = ['\'relu\'', '\'tanh\'', '\'sigmoid\'', '\'hard_sigmoid\'', '\'softmax\'', '\'elu\'', '\'selu\'',
# '\'softplus\'', '\'softsign\'', '\'linear\'']
bools = [True, False]
dropouts = np.linspace(0, 1, num=DROPOUTS_N_SAMPLES, endpoint=False)

# Options for each layer type

LSTM_options = {  # Maps parameters to trail values
    'units': units,
    'activation': activations,
    'recurrent_activation': activations,
    'use_bias': bools,
    'dropout': dropouts,
    'recurrent_dropout': dropouts,
    'return_sequences': [True]
    # Not using all availible options for now
    # 'kernel_initializer': None,
    # 'recurrent_initializer': None,
    # 'bias_initializer': None,
    # 'unit_forget_bias': None,
    # 'kernel_regularizer': None,
    # 'recurrent_regularizer': None,
    # 'bias_regularizer': None,
    # 'activity_regularizer': None,
    # 'kernel_constraint': None,
    # 'recurrent_constraint': None,
    # 'bias_constraint': None,
}

layer_options = {
    'LSTM': LSTM_options
}

compiler_options = {

}


def gen_layer_params(layer_options):  # TODO: Add non-random generation
    return ', '.join([str(param) + '=' + str(random.choice(layer_options[param])) for param in layer_options.keys()])


def gen_layer(layer_name, layer_options):
    return eval('{0}({1})'.format(layer_name, gen_layer_params(layer_options)))


def gen_model(input_layer, hidden_layers, output_layer, compile_args):
    model = Sequential()
    # Input layer
    model.add(input_layer)
    # Hidden layers
    for i in range(0, len(hidden_layers)):
        model.add(hidden_layers[i])
    # Output layer
    model.add(output_layer)
    # Compilation
    model.compile()
    return model

model = Sequential()
model.add(LSTM(16,
               return_sequences=True,
               input_shape=(16, 1)))
model.add(gen_layer('LSTM', LSTM_options))
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


print('Imported cnv_net')


'''
# Options for layers
# Use a dictionary of dictionary to specify params
# Can try to programmatically set 

# LSTM
model_1.add(LSTM(
    units=1,  # Required parameter
    activation='tanh',  # Optional parameters with defaults
    recurrent_activation='hard_sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0,
    recurrent_dropout=0))
    
    
print('LSTM({layer_width}, return_sequences={return_sequences}, input_shape=({timesteps}, {in_dim})'.format(
    layer_width=4, return_sequences=True, timesteps=16, in_dim=1)))


ret_seq = True
gen_model(['LSTM({layer_width}, return_sequences={return_sequences}, input_shape=({timesteps}, {in_dim})'.format(
    layer_width=4, return_sequences=True, timesteps=16, in_dim=1),
           'LSTM(4, return_sequences=True, activation=\'softmax\''],
          'optimizer=\'rmsprop\', loss=\'binary_crossentropy\', metrics=[\'accuracy\']')
    
'''
