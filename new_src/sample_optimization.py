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
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
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
        train_score  = numpy.sum(train_losses)
        valid_score  = numpy.sum(valid_losses)
        print 'epoch %i, train_score %f, valid_score %f' % (epoch, float(train_score) / train_sample_num, float(valid_score) / valid_sample_num)
        '''
        train_score  = numpy.mean(train_losses)
        valid_score  = numpy.mean(valid_losses)
        print 'epoch %i, train_score %f, valid_score %f' % (epoch, train_score, valid_score)
        '''


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python %s (dataset_file)' % (sys.argv[0])
        sys.exit()
    sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset=sys.argv[1], batch_size=600)

