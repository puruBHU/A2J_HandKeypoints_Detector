# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:10:51 2021

@author: Purnendu Mishra
"""

import numpy as np

import tensorflow as tf
from   tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
from   tensorflow.keras.layers import Layer, InputSpec


def spatial_softmax(x):
    y = tf.exp(x - tf.reduce_max(x, axis=(1,2), keepdims=True))
    y = y / tf.reduce_sum(y, axis= (1,2), keepdims=True)
    return y
    

class SpatialSoftmax(Layer):
    def __init__(self, name):
        super(SpatialSoftmax, self).__init__(name = name)
        self.scale        = 1.0
        
    def call(self, x):
        
        output = spatial_softmax(x) 
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape


class SpatialSoftmaxV2(Layer):
    def __init__(self, **kwargs):
        super(SpatialSoftmaxV2, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     self.input_spec = [InputSpec(shape=input_shape)] 
        
    #     initializer     = RandomNormal(stddev = 0.01)
        
    #     self.K          = self.add_weight(shape       = (input_shape[3], 
    #                                                      ),
    #                                       initializer = initializer,
    #                                       trainable   = True)
        
    #     super(SpatialSoftmaxV2, self).build(input_shape)
        
        
    def call(self, x):
        z = x /self.K
        
        B,H,W,C = z.shape
        # print(B,H,W,C)

        tr        = tf.transpose(z, [0,3,1,2])
        re        = tf.reshape(tr, [-1, H*W])
        
        softmax   = tf.nn.softmax(re, axis = -1)
        
        reshape   = tf.reshape(softmax, [-1, C, H, W])
        output    = tf.transpose(reshape, [0,2,3,1])
        
        return output
        
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class KeypointsRegression(Layer):
    def __init__(self, **kwargs):
        super(KeypointsRegression, self).__init__(**kwargs)
        
    def call(self,x):
        
        ap = x[0]
        os = x[1]
    
        _, H, W, K = ap.shape
        
        i = np.arange(H) / H
        j = np.arange(W) / W
        
        coords       = np.meshgrid(i,j, indexing = 'ij')
        image_coords = np.array(coords, dtype = np.float32).T
        kernel       = np.tile(image_coords[:,:,None,:],[1,1,K,1])
        
        kernel       = tf.constant(kernel, dtype = tf.float32) 
        pos          = tf.math.add(kernel, os)  # Position of keypoint plus offsets
        
        ap  = tf.expand_dims(ap, axis = -1)
        # Multiply anchor proposal and offsets
        mul  = ap * pos
        
        keys = tf.reduce_sum(mul, axis = (1,2))
        return keys
        
    def comput_output_shape(self, x):
        B, H,W,K,C = x[1].shape
        return (B,K,C)