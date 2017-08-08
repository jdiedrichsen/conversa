''' cnv_net - Provides neural net utilities for Conversa '''

import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, GRU, LSTM
from keras.layers.convolutional import Conv1D, Conv2D

__author__ = 'Shayaan Syed Ali'
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ['Shayaan Syed Ali']
__maintainer__ = 'Shayaan Syed Ali'
__email__ = 'shayaan.syed.ali@gmail.com'
__status__ = 'Development'
# __license__ = ''
# __version__ = ''


# Sample input:
# gen_model([
# 'LSTM(4, return_sequences=True, input_shape=(16, 1)',
# 'LSTM(4, return_sequences=True, activation=\'softmax\''],
# 'optimizer=\'rmsprop\', loss=\'binary_crossentropy\', metrics=[\'accuracy\']')
def gen_model(input_layer, hidden_layers, compile_args):
    model = Sequential()
    # Input layer
    model.add()
    # Hidden layers

    # Output layer

    # Compilation
    model.compile(eval(compile_args))
    return model

# print('LSTM({layer_width}, return_sequences={return_sequences}, input_shape=({timesteps}, {in_dim})'.format(
#     layer_width=4, return_sequences=True, timesteps=16, in_dim=1))

ret_seq = True
gen_model(['LSTM({layer_width}, return_sequences={return_sequences}, input_shape=({timesteps}, {in_dim})'.format(
    layer_width=4, return_sequences=True, timesteps=16, in_dim=1),
           'LSTM(4, return_sequences=True, activation=\'softmax\''],
          'optimizer=\'rmsprop\', loss=\'binary_crossentropy\', metrics=[\'accuracy\']')

exit(0)

'''
# Options for layers
# Use a dictionary of dictionary to specify params

# LSTM
model.add(LSTM(
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
    
'''