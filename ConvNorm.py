from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
import numpy as np


class ConvNorm(Layer):

    def __init__(self, filters, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', bias_initializer='zeros', group_size=0, **kwargs):
        super(ConvNorm, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = strides
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        if group_size>0:
            assert np.mod(filters, group_size)==0
        self.group_size = group_size

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        if self.group_size>0:
            assert np.mod(input_dim, self.group_size)==0
            input_dim = self.group_size
        
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer, name='kernel')
        self.bias = self.add_weight(shape=[1, 1, 1, self.filters], initializer=self.bias_initializer, name='bias')
        self.built = True

    def call(self, inputs):
        weights = self.kernel
        
        x = inputs

        #Normalize
        d = K.sqrt(K.sum(K.square(weights), axis=[0,1,2], keepdims=True) + 1e-8)
        weights = weights / d
        
        if self.kernel_size[0]>1:
            p = (self.kernel_size[0]-1)//2
            x = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode='SYMMETRIC')
        
        x = tf.nn.conv2d(x, weights, strides=self.strides, padding="VALID")
        x = x + self.bias
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape[:3] + self.out_channels
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvNorm_freq(Layer):

    def __init__(self, filters, kernel_size, strides=1, kernel_initializer='glorot_uniform', bias_initializer='zeros', group_size=0, **kwargs):
        super(ConvNorm_freq, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        if group_size>0:
            assert np.mod(filters, group_size)==0
        self.group_size = group_size

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        if self.group_size>0:
            assert np.mod(input_dim, self.group_size)==0
            input_dim = self.group_size
        
        kernel_shape = (input_shape[1], self.kernel_size, input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer, name='kernel')
        self.bias = self.add_weight(shape=[1, 1, 1, self.filters], initializer=self.bias_initializer, name='bias')
        self.built = True

    def call(self, inputs):
        weights = self.kernel
        
        x = inputs

        #Normalize
        d = K.sqrt(K.sum(K.square(weights), axis=[0,1,2], keepdims=True) + 1e-8)
        weights = weights / d
        
        if self.kernel_size>1:
            p = (self.kernel_size-1)//2
            x = tf.pad(x, [[0,0], [0,0], [p,p], [0,0]], mode='SYMMETRIC')
        
        x = tf.nn.conv2d(x, weights, strides=self.strides, padding="VALID")
        x = x + self.bias
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape[:3] + self.out_channels
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvNorm_freq, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))