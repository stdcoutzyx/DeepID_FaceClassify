#!/usr/bin/env python
# -*- coding:utf8 -*-

import cPickle
import pickle
import gzip
import os
import sys

import theano
import theano.tensor as T
import numpy

def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set = cPickle.load(f)[0:2]
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
                numpy.asarray(data_x, dtype=theano.config.floatX),
                borrow=borrow)
        shared_y = theano.shared(
                numpy.asarray(data_y, dtype=theano.config.floatX),
                borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]

def load_data_nogzip(dataset):
    f = open(dataset, 'rb')
    train_set, valid_set = pickle.load(f)[0:2]
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
                numpy.asarray(data_x, dtype=theano.config.floatX),
                borrow=borrow)
        shared_y = theano.shared(
                numpy.asarray(data_y, dtype=theano.config.floatX),
                borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
