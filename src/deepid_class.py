#!/usr/bin/env python
# -*- coding:utf8 -*-

from layers import *
from load_data import *

import cPickle
import gzip
import os
import sys
import time
import numpy
import theano

class DeepID:
    def __init__(self):
        self.rng = numpy.random.RandomState(1234)

    def load_data_deepid(self, dataset='', batch_size=500):
        print 'loading data ...'
        datasets = load_data_nogzip(dataset)
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]

        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / batch_size
        self.batch_size = batch_size

    def layer_params(self, nkerns=[20,40,60,80]):
        src_channel = 3
        self.layer1_image_shape  = (self.batch_size, src_channel, 55, 47)
        self.layer1_filter_shape = (nkerns[0],  src_channel, 4, 4)
        self.layer2_image_shape  = (self.batch_size, nkerns[0], 26, 22)
        self.layer2_filter_shape = (nkerns[1], nkerns[0], 3, 3)
        self.layer3_image_shape  = (self.batch_size, nkerns[1], 12, 10)
        self.layer3_filter_shape = (nkerns[2], nkerns[1], 3, 3)
        self.layer4_image_shape  = (self.batch_size, nkerns[2], 5, 4)
        self.layer4_filter_shape = (nkerns[3], nkerns[2], 2, 2)
        self.result_image_shape  = (self.batch_size, nkerns[3], 4, 3)

    def build_layer_architecture(self, n_hidden=160, n_out=100, acti_func=relu):
        '''
        simple means the deepid layer input is only the layer4 output.
        layer1: convpool layer
        layer2: convpool layer
        layer3: convpool layer
        layer4: conv layer
        deepid: hidden layer
        softmax: logistic layer
        '''
        self.index      = T.lscalar()
        self.step_rate  = T.dscalar()
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        print 'building the model ...'

        layer1_input = self.x.reshape(self.layer1_image_shape)
        layer1 = LeNetConvPoolLayer(self.rng,
                input        = layer1_input,
                image_shape  = self.layer1_image_shape,
                filter_shape = self.layer1_filter_shape,
                poolsize     = (2,2),
                activation   = acti_func)

        layer2 = LeNetConvPoolLayer(self.rng,
                input        = layer1.output,
                image_shape  = self.layer2_image_shape,
                filter_shape = self.layer2_filter_shape,
                poolsize     = (2,2),
                activation   = acti_func)

        layer3 = LeNetConvPoolLayer(self.rng,
                input        = layer2.output,
                image_shape  = self.layer3_image_shape,
                filter_shape = self.layer3_filter_shape,
                poolsize     = (2,2),
                activation   = acti_func)

        layer4 = LeNetConvLayer(self.rng,
                input        = layer3.output,
                image_shape  = self.layer4_image_shape,
                filter_shape = self.layer4_filter_shape,
                activation   = acti_func)

        # deepid_input = layer4.output.flatten(2)

        layer3_output_flatten = layer3.output.flatten(2)
        layer4_output_flatten = layer4.output.flatten(2)
        deepid_input = T.concatenate([layer3_output_flatten, layer4_output_flatten], axis=1)

        deepid_layer = HiddenLayer(self.rng,
                input = deepid_input,
                n_in  = numpy.prod( self.result_image_shape[1:] ) + numpy.prod( self.layer4_image_shape[1:] ),
                # n_in  = numpy.prod( self.result_image_shape[1:] ),
                n_out = n_hidden,
                activation = acti_func)
        self.softmax_layer = LogisticRegression(
                input = deepid_layer.output,
                n_in = n_hidden,
                n_out = n_out)

        self.cost   = self.softmax_layer.negative_log_likelihood(self.y)
        self.params = self.softmax_layer.params + deepid_layer.params + layer4.params + layer3.params + layer2.params + layer1.params
    
    def build_test_valid_model(self):
        self.test_valid_model = theano.function(inputs=[self.index],
                outputs=self.softmax_layer.errors(self.y),
                givens = {
                    self.x: self.valid_set_x[self.index * self.batch_size : (self.index+1) * self.batch_size],
                    self.y: self.valid_set_y[self.index * self.batch_size : (self.index+1) * self.batch_size]}
                )

    def build_test_train_model(self):
        self.test_train_model = theano.function(inputs=[self.index],
                outputs=self.softmax_layer.errors(self.y),
                givens = {
                    self.x: self.train_set_x[self.index * self.batch_size : (self.index+1) * self.batch_size],
                    self.y: self.train_set_y[self.index * self.batch_size : (self.index+1) * self.batch_size]}
                )

    def build_train_model(self):
        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.step_rate * gparam))

        self.train_model = theano.function(inputs=[self.index, self.step_rate],
                outputs=self.cost,
                updates=updates,
                givens = {
                    self.x: self.train_set_x[self.index * self.batch_size : (self.index+1) * self.batch_size],
                    self.y: self.train_set_y[self.index * self.batch_size : (self.index+1) * self.batch_size]}
                )

    def train(self, n_epochs=200, learning_rate=0.01):
        print 'Train the model ...'
        train_sample_num = self.train_set_x.get_value(borrow=True).shape[0]
        valid_sample_num = self.valid_set_x.get_value(borrow=True).shape[0]

        epoch = 0
        while epoch < n_epochs:
            epoch += 1
            for minibatch_index in xrange( self.n_train_batches ):
                minibatch_cost = self.train_model(minibatch_index, learning_rate)
                print '\tepoch %i, minibatch_index %i/%i, minibatch_cost %f' % (epoch, minibatch_index, self.n_train_batches, minibatch_cost)
            # train_losses = [test_train_model(i) for i in xrange(n_train_batches)]
            valid_losses = [self.test_valid_model(i) for i in xrange( self.n_valid_batches) ]

            '''
            train_score  = numpy.sum(train_losses)
            valid_score  = numpy.sum(valid_losses)
            print 'epoch %i, train_score %f, valid_score %f' % (epoch, float(train_score) / train_sample_num, float(valid_score) / valid_sample_num)
            '''
            # train_score  = numpy.mean(train_losses)
            valid_score  = numpy.mean(valid_losses)
            print 'epoch %i, train_score %f, valid_score %f' % (epoch, 100., valid_score)

def simple_deepid(learning_rate=0.1, n_epochs=200, dataset='',
        nkerns=[20,40,60,80], batch_size=500, n_hidden=160, n_out=100, acti_func=relu):
    deepid = DeepID()
    deepid.load_data_deepid(dataset, batch_size)
    deepid.layer_params(nkerns)
    deepid.build_layer_architecture(n_hidden, n_out, acti_func)
    deepid.build_test_train_model()
    deepid.build_test_valid_model()
    deepid.build_train_model()
    deepid.train(n_epochs, learning_rate)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python %s (dataset)' % (sys.argv[0])
        sys.exit()
    simple_deepid(learning_rate=0.01, n_epochs=200, dataset=sys.argv[1], nkerns=[20,40,60,80], batch_size=500, n_hidden=160, n_out=1357, acti_func=relu)
