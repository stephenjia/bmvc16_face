# -*- coding: utf-8 -*-


import numpy as np
from collections import OrderedDict
import sys
import os

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers.base import Layer
from lasagne.layers import MergeLayer

from lasagne.layers.conv import conv_output_length
from lasagne.layers.pool import pool_output_length
from lasagne.utils import as_tuple
from lasagne import utils

__all__ = [
    "Unpool2DLayer"
]
        
class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    ds is a tuple, denotes the upsampling
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        """
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)
        """
        shp = input.shape
        upsample = T.zeros((shp[0], shp[1], shp[2] * 2, shp[3] * 2), dtype=input.dtype)
        
        # top-left unpooling
        upsample = T.set_subtensor(upsample[:, :, ::ds[0], ::ds[1]], input)         

        return upsample
