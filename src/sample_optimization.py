#!/usr/bin/env python
# -*- coding:utf8 -*-

import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T

from layers import *
from load_data import *

def sgd_optimization_mnist(learning_rate=0.13, 
        n_epochs=1000, dataset='mnist.pkl.gz', batch_size=300):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    step_rate = T.dscalar()

    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    test_valid_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens = {
                x: valid_set_x[index * batch_size : (index+1) * batch_size],
                y: valid_set_y[index * batch_size : (index+1) * batch_size]}
            )

    test_train_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens = {
                x: train_set_x[index * batch_size : (index+1) * batch_size],
                y: train_set_y[index * batch_size : (index+1) * batch_size]}
            )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - step_rate * g_W),
               (classifier.b, classifier.b - step_rate * g_b)]
    train_model = theano.function(inputs=[index, step_rate],
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
        if epoch > 50:
            learning_rate = 0.1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index, learning_rate)
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

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, 
        n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    classifier = MLP(rng=rng, input=x, n_in=28*28,
                     n_hidden=n_hidden, n_out=10)
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    test_valid_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens = {
                x: valid_set_x[index * batch_size : (index+1) * batch_size],
                y: valid_set_y[index * batch_size : (index+1) * batch_size]}
            )

    test_train_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens = {
                x: train_set_x[index * batch_size : (index+1) * batch_size],
                y: train_set_y[index * batch_size : (index+1) * batch_size]}
            )

    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(classifier.params, gparams):
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


def evaluate_lenet3(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20,50], batch_size=500):
    '''
    layer0: convpool layer
    layer1: convpool layer
    layer1: hidden layer
    layer2: logistic layer
    '''

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    image_shape = (batch_size, 1, 28, 28)
    rng = numpy.random.RandomState(1234)

    print 'building the model ...'

    layer0_input = x.reshape(image_shape)
    layer0 = LeNetConvPoolLayer(rng, 
            input        = layer0_input,
            image_shape  = image_shape,
            filter_shape = (nkerns[0], 1, 5, 5),
            poolsize     = (2, 2),
			activation	 = relu)
    layer1 = LeNetConvPoolLayer(rng,
            input        = layer0.output,
            image_shape  = (batch_size, nkerns[0], 12, 12),
            filter_shape = (nkerns[1],  nkerns[0], 5, 5),
            poolsize     = (2,2),
			activation	 = relu)

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(rng, 
            input   = layer2_input, 
            n_in    = nkerns[1] * 4 * 4,
            n_out   = 500, 
            activation = relu)
    layer3 = LogisticRegression(
            input   = layer2.output,
            n_in    = 500,
            n_out   = 10)

    cost = layer3.negative_log_likelihood(y)

    test_valid_model = theano.function(inputs=[index],
            outputs=layer3.errors(y),
            givens = {
                x: valid_set_x[index * batch_size : (index+1) * batch_size],
                y: valid_set_y[index * batch_size : (index+1) * batch_size]}
            )

    test_train_model = theano.function(inputs=[index],
            outputs=layer3.errors(y),
            givens = {
                x: train_set_x[index * batch_size : (index+1) * batch_size],
                y: train_set_y[index * batch_size : (index+1) * batch_size]}
            )

    params = layer3.params + layer2.params + layer1.params + layer0.params
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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python %s (dataset_file)' % (sys.argv[0])
        sys.exit()
    sgd_optimization_mnist(learning_rate=0.2, n_epochs=1000, dataset=sys.argv[1], batch_size=600)
    # test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset=sys.argv[1], batch_size=20, n_hidden=500)
    # evaluate_lenet3(learning_rate=0.1, n_epochs=200, dataset=sys.argv[1], nkerns=[20, 50], batch_size=500)
