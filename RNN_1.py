from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers, activations
from tensorflow.keras import backend as K

class RNNCell(keras.layers.Layer):
    def __init__(self, units, mask_matrix, dt = 0.1, tau = 10, noise_var = 0.1, dale_ratio= None,
                 activation = 'relu', W_in_reg = False, W_r_reg = False, **kwargs):
        self.dt = dt
        self.tau = tau
        self.state_size = units  #number of RNN cells
        self.output_size = units
        self.noise = noise_var
        self.dale_ratio = dale_ratio

        if self.dale_ratio:
            dale_vec = np.ones(self.state_size)
            dale_vec[int(dale_ratio*self.state_size):] = -1
            dale_matrix = np.diag(dale_vec)
            self.D = K.variable(dale_matrix)

        self.mask = K.variable(mask_matrix)

        self.activation = activations.get(activation)
        self.W_in_reg = W_in_reg
        self.W_r_reg = W_r_reg

        super(RNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.W_in_reg:
            self.W_in = self.add_weight(shape=(input_shape[-1], self.state_size),
                                    initializer='glorot_normal', regularizer=regularizers.l1(1e-8),
                                    name='input_weights')
        else:
            self.W_in = self.add_weight(shape=(input_shape[-1], self.state_size),
                                        initializer='glorot_normal',
                                        name='input_weights')

        if self.W_r_reg:
            self.W_r = self.add_weight(shape=(self.state_size, self.state_size),
                                   initializer='glorot_normal', regularizer=regularizers.l2(1e-5),
                                   name='recurrent_weights')
        else:
            self.W_r = self.add_weight(shape=(self.state_size, self.state_size),
                                       initializer='glorot_normal',
                                       name='recurrent_weights')

        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        if self.dale_ratio:
            h_in = K.dot(inputs, K.abs(self.W_in))
            h_rec = K.dot(self.activation(prev_output), self.mask*K.abs(self.W_r)*self.D)
        else:
            h_in = K.dot(inputs, self.W_in)
            h_rec = tf.matmul(self.activation(prev_output), self.mask*self.W_r)

        alpha = self.dt/self.tau
        output = (1-alpha)*prev_output+alpha*(h_rec+h_in)+\
                 K.random_normal(shape=(self.state_size,), mean = 0.0, stddev=self.noise)

        return output, [output]

# cell = RNNCell(units=8, dt = 0.1, tau = 1, dale_ratio=0.8, W_in_reg='l2', W_r_reg='l2')
# x = keras.Input((None, 5))
# layer = keras.layers.RNN(cell, return_sequences=True)
# y = layer(x)
# model = tf.keras.Model(x,y)
# model.summary()


