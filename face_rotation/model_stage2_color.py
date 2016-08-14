import numpy as np
import theano
import theano.tensor as T

import sys
import os
import time
import socket
import cPickle as pickle

# xu
from layers.NewLayer import Unpool2DLayer

# lasagne
import lasagne
from lasagne.layers import DenseLayer, ReshapeLayer, ConcatLayer, Gate, Conv2DLayer, MaxPool2DLayer, DropoutLayer, SliceLayer, InputLayer, DimshuffleLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.init import HeNormal, Orthogonal, Normal, Uniform
from lasagne.nonlinearities import softmax, identity, sigmoid, tanh, rectify, LeakyRectify
leaky_rectify = LeakyRectify(0.2)
from lasagne.updates import rmsprop, adam, sgd, apply_momentum

class Model(object):
    """ model initialization """
    def __init__(self, options):
        self.options = options        
        
    """ build and compile model """
    def build_model(self, img_batch, img_batch_gen):
                
        img_size = self.options['img_size']
        pose_code_size = self.options['pose_code_size']                        
        filter_size = self.options['filter_size']        
        batch_size = img_batch.shape[0]
        
        # image encoding               
        l_in_1 = InputLayer(shape = [None, img_size[0], img_size[1], img_size[2]], input_var=img_batch)
        l_in_1_dimshuffle = DimshuffleLayer(l_in_1, (0,3,1,2))        
        l_in_2 = InputLayer(shape = [None, img_size[0], img_size[1], img_size[2]], input_var=img_batch_gen)
        l_in_2_dimshuffle = DimshuffleLayer(l_in_2, (0,3,1,2)) 
        l_in_concat = ConcatLayer([l_in_1_dimshuffle, l_in_2_dimshuffle], axis=1)                         
        
        l_conv1_1 = Conv2DLayer(l_in_concat, num_filters=64, filter_size=filter_size, W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))        
        l_conv1_2 = Conv2DLayer(l_conv1_1, num_filters=64, filter_size=filter_size, W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))
        l_pool1 = MaxPool2DLayer(l_conv1_2, pool_size=(2,2)) 
        
        l_conv2_1 = Conv2DLayer(l_pool1, num_filters=128, filter_size=filter_size, nonlinearity=rectify,W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))
        l_conv2_2 = Conv2DLayer(l_conv2_1, num_filters=128, filter_size=filter_size, nonlinearity=rectify,W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))
        
        l_pool2 = MaxPool2DLayer(l_conv2_2, pool_size=(2,2))         
        l_conv_3 = Conv2DLayer(l_pool2, num_filters=128, filter_size=(1,1), W=HeNormal())
        l_unpool1 = Unpool2DLayer(l_conv_3, ds = (2,2))        
        
        # image decoding
        l_deconv_conv1_1 = Conv2DLayer(l_unpool1, num_filters=128, filter_size=filter_size, nonlinearity=rectify,W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))
        l_deconv_conv1_2 = Conv2DLayer(l_deconv_conv1_1, num_filters=64, filter_size=filter_size, nonlinearity=rectify,W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))
        
        l_unpool2 = Unpool2DLayer(l_deconv_conv1_2, ds = (2,2))
        l_deconv_conv2_1 = Conv2DLayer(l_unpool2, num_filters=64, filter_size=filter_size, nonlinearity=None, W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))
        l_deconv_conv2_2 = Conv2DLayer(l_deconv_conv2_1, num_filters=img_size[2], filter_size=filter_size, nonlinearity=None, W=HeNormal(), pad=(filter_size[0]//2, filter_size[1]//2))  
                
        return l_deconv_conv2_2                
        
        
        
 

