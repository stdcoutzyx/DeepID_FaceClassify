#!/usr/bin/env python
# -*- coding:utf8 -*-

from layer import *
from load_data import *

import cPickle
import gzip
import os
import sys
import time
import numpy
import theano

def simple_deepid(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', 
        nkerns=[20,40,60,80], batch_size=500, n_hidden=160, n_out=100, acti_func=relu):
    '''
    simple means the deepid layer input is only the layer4 output.
    layer1: convpool layer
    layer2: convpool layer
    layer3: convpool layer
    layer4: conv layer
    deepid: hidden layer
    softmax: logistic layer
    '''

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    src_channel = 1
    layer1_image_shape  = (batch_size, src_channel, 39, 31)
    layer1_filter_shape = (nkerns[0],  src_channel, 4, 4)
    layer2_image_shape  = (batch_size, nkerns[0], 18, 14)
    layer2_filter_shape = (nkerns[1], nkerns[0], 3, 3)
    layer3_image_shape  = (batch_size, nkerns[1], 8, 6)
    layer3_filter_shape = (nkerns[2], nkerns[1], 3, 3)
    layer4_image_shape  = (batch_size, nkerns[2], 3, 2)
    layer4_filter_shape = (nkerns[3], nkerns[2], 2, 2)
    result_image_shape  = (batch_size, nkerns[3], 2, 1)

    rng = numpy.random.RandomState(1234)

    print 'building the model ...'

    layer1_input = x.reshape(layer1_image_shape)
    layer1 = LeNetConvPoolLayer(rng,
            input        = layer1_input,
            image_shape  = layer1_image_shape,
            filter_shape = layer1_filter_shape,
            poolsize     = (2,2),
            activation   = acti_func)

    layer2 = LeNetConvPoolLayer(rng,
            input        = layer1.output,
            image_shape  = layer2_image_shape,
            filter_shape = layer2_filter_shape,
            poolsize     = (2,2),
            activation   = acti_func)

    layer3 = LeNetConvPoolLayer(rng,
            input        = layer2.output,
            image_shape  = layer3_image_shape,
            filter_shape = layer3_filter_shape,
            poolsize     = (2,2),
            activation   = acti_func)

    layer4 = LeNetConvLayer(rng,
            input        = layer3.output,
            image_shape  = layer4_image_shape,
            filter_shape = layer4_filter_shape,
            activation   = acti_func)

    deepid_input = layer4.output.flatten(2)

    '''
    layer3_output_flatten = layer3.output.flatten(2)
    layer4_output_flatten = layer4.output.flatten(2)
    deepid_input = T.concatename([layer3_output_flatten, layer4_output_flatten], axis=1)
    '''

    deepid_layer = HiddenLayer(rng,
            input = deepid_input,
            n_in  = numpy.prod(result_image_shape[1:]),
            n_out = n_hidden,
            activation = acti_func)
    softmax_layer = LogisticRegression(
            input = deepid_layer.output,
            n_in = n_hidden,
            n_out = n_out)


    cost = softmax_layer.negative_log_likelihood(y)

    test_valid_model = theano.function(inputs=[index],
            outputs=softmax_layer.errors(y),
            givens = {
                x: valid_set_x[index * batch_size : (index+1) * batch_size],
                y: valid_set_y[index * batch_size : (index+1) * batch_size]}
            )

    test_train_model = theano.function(inputs=[index],
            outputs=softmax_layer.errors(y),
            givens = {
                x: train_set_x[index * batch_size : (index+1) * batch_size],
                y: train_set_y[index * batch_size : (index+1) * batch_size]}
            )

    params = softmax_layer.params + deepid_layer.params + layer4.params + layer3.params + layer2.params + layer1.params
    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens = {
                x: train_set_x[index * batch_size : (index+1) * batch_size],
                y: train_set_y[index * batch_size : (index+1) * batch_size]}
            )
    print 'Train the model ...'
    train_sample_num = train_set_x.get_value(borrow=True).shape[0]
    valid_sample_num = valid_set_x.get_value(borrow=True).shape[0]

    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
            print '\tepoch %i, minibatch_index %i/%i, minibatch_cost %f' % (epoch, minibatch_index, n_train_batches, minibatch_cost)
        train_losses = [test_train_model(i) for i in xrange(n_train_batches)]
        valid_losses = [test_valid_model(i) for i in xrange(n_valid_batches)]

        '''
        train_score  = numpy.sum(train_losses)
        valid_score  = numpy.sum(valid_losses)
        print 'epoch %i, train_score %f, valid_score %f' % (epoch, float(train_score) / train_sample_num, float(valid_score) / valid_sample_num)
        '''
        train_score  = numpy.mean(train_losses)
        valid_score  = numpy.mean(valid_losses)
        print 'epoch %i, train_score %f, valid_score %f' % (epoch, train_score, valid_score)
