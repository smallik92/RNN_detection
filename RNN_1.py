from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers, activations, initializers
from tensorflow.keras import backend as K

class RNNCell(keras.layers.Layer):
    def __init__(self, units, mask_matrix, dt = 0.1, tau = 10, noise_var = 0.1, dale_ratio= None,
                 activation = 'relu', **kwargs):
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
        self.input_init = initializers.RandomUniform(minval=0., maxval=0.01, seed=1234)
        self.rec_init = initializers.RandomUniform(minval=0., maxval=0.05, seed=1234)

        super(RNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_in = self.add_weight(shape=(input_shape[-1], self.state_size),
                                        initializer=self.input_init,
                                        name='input_weights')
        self.W_r = self.add_weight(shape=(self.state_size, self.state_size),
                                       initializer=self.rec_init,
                                       name='recurrent_weights')

        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        if self.dale_ratio:
            h_in = K.dot(inputs, K.abs(self.W_in))
            h_rec = K.dot(self.activation(prev_output), self.mask*K.dot(K.abs(self.W_r),self.D))
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


