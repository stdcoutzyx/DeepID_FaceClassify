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

class ParamDumpHelper:
    def __init__(self, dump_file):
        self.dump_file = dump_file

    def dump(self, params):
        dumped_params = self.get_params_from_file()
        dumped_params.append(params)
        self.params_to_file(dumped_params)

    def params_to_file(self, params):
        f = gzip.open(self.dump_file, 'wb')
        if len(params) > 20:
            params = params[10:]
        pickle.dump(params, f)
        f.close()

    def get_params_from_file(self):
        if os.path.exists(self.dump_file):
            f = gzip.open(self.dump_file, 'rb')
            dumped_params = pickle.load(f)
            f.close()
            return dumped_params
        return []
        
class DeepID:
    def __init__(self, pd_helper):
        self.rng = numpy.random.RandomState(1234)
        self.pd_helper = pd_helper
        exist_params = pd_helper.get_params_from_file()
        if len(exist_params) != 0:
            self.exist_params = exist_params[-1]
        else:
            self.exist_params = [[None, None],
                                 [None, None],
                                 [None, None],
                                 [None, None],
                                 [None, None],
                                 [None, None],
                                 1.0,
                                 1.0,
                                 0]

    def load_data_deepid(self, dataset_file, batch_size):
        print 'loading data ...'
        datasets = load_data_split_pickle(dataset_file)
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]

        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / batch_size
        self.batch_size = batch_size

        print 'train_x: ', self.train_set_x.get_value(borrow=True).shape
        print 'train_y: ', self.train_set_y.shape
        print 'valid_x: ', self.valid_set_x.get_value(borrow=True).shape
        print 'valid_y: ', self.valid_set_y.shape


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

    def build_layer_architecture(self, n_hidden, n_out, acti_func=relu):
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
        self.layer1 = LeNetConvPoolLayer(self.rng,
                input        = layer1_input,
                image_shape  = self.layer1_image_shape,
                filter_shape = self.layer1_filter_shape,
                poolsize     = (2,2),
                W = self.exist_params[5][0],
                b = self.exist_params[5][1],
                activation   = acti_func)

        self.layer2 = LeNetConvPoolLayer(self.rng,
                input        = self.layer1.output,
                image_shape  = self.layer2_image_shape,
                filter_shape = self.layer2_filter_shape,
                poolsize     = (2,2),
                W = self.exist_params[4][0],
                b = self.exist_params[4][1],
                activation   = acti_func)

        self.layer3 = LeNetConvPoolLayer(self.rng,
                input        = self.layer2.output,
                image_shape  = self.layer3_image_shape,
                filter_shape = self.layer3_filter_shape,
                poolsize     = (2,2),
                W = self.exist_params[3][0],
                b = self.exist_params[3][1],
                activation   = acti_func)

        self.layer4 = LeNetConvLayer(self.rng,
                input        = self.layer3.output,
                image_shape  = self.layer4_image_shape,
                filter_shape = self.layer4_filter_shape,
                W = self.exist_params[2][0],
                b = self.exist_params[2][1],
                activation   = acti_func)

        # deepid_input = layer4.output.flatten(2)

        layer3_output_flatten = self.layer3.output.flatten(2)
        layer4_output_flatten = self.layer4.output.flatten(2)
        deepid_input = T.concatenate([layer3_output_flatten, layer4_output_flatten], axis=1)

        self.deepid_layer = HiddenLayer(self.rng,
                input = deepid_input,
                n_in  = numpy.prod( self.result_image_shape[1:] ) + numpy.prod( self.layer4_image_shape[1:] ),
                # n_in  = numpy.prod( self.result_image_shape[1:] ),
                n_out = n_hidden,
                W = self.exist_params[1][0],
                b = self.exist_params[1][1],
                activation = acti_func)
        self.softmax_layer = LogisticRegression(
                input = self.deepid_layer.output,
                n_in  = n_hidden,
                n_out = n_out,
                W = self.exist_params[0][0],
                b = self.exist_params[0][1])

        self.cost = self.softmax_layer.negative_log_likelihood(self.y)
    
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
        self.params = self.softmax_layer.params + self.deepid_layer.params + self.layer4.params \
                + self.layer3.params + self.layer2.params + self.layer1.params
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

    def train(self, n_epochs, learning_rate):
        print 'Training the model ...'
        train_sample_num = self.train_set_x.get_value(borrow=True).shape[0]
        valid_sample_num = self.valid_set_x.get_value(borrow=True).shape[0]

        loss_records = []

        epoch = self.exist_params[-1]
        while epoch < n_epochs:
            train_losses = []
            for minibatch_index in xrange( self.n_train_batches ):
                minibatch_cost = self.train_model(minibatch_index, learning_rate)
                train_loss     = self.test_train_model(minibatch_index)
                train_losses.append(train_loss)

                line = '\r\tepoch %i, minibatch_index %i/%i, error %f' % (epoch, minibatch_index, self.n_train_batches, train_loss)
                sys.stdout.write(line)
                sys.stdout.flush()

            valid_losses = [self.test_valid_model(i) for i in xrange( self.n_valid_batches) ]

            train_score = numpy.mean(train_losses)
            valid_score = numpy.mean(valid_losses)
            loss_records.append((epoch, train_score, valid_score))
            print '\nepoch %i, train_score %f, valid_score %f' % (epoch, train_score, valid_score)

            params = [self.softmax_layer.params, 
                      self.deepid_layer.params, 
                      self.layer4.params,
                      self.layer3.params, 
                      self.layer2.params,
                      self.layer1.params,
                      valid_score,
                      train_score,
                      epoch]
            self.pd_helper.dump(params)
            epoch += 1
        return loss_records


def simple_deepid(learning_rate, n_epochs, dataset, params_file,
        nkerns, batch_size, n_hidden, n_out, acti_func):
    pd_helper = ParamDumpHelper(params_file)
    deepid = DeepID(pd_helper)
    deepid.load_data_deepid(dataset, batch_size)
    deepid.layer_params(nkerns)
    deepid.build_layer_architecture(n_hidden, n_out, acti_func)
    deepid.build_test_train_model()
    deepid.build_test_valid_model()
    deepid.build_train_model()
    loss_records = deepid.train(n_epochs, learning_rate)
    
    print ''
    for record in loss_records:
        print record[0], record[1], record[2]
    

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python %s vec_valid vec_train params_file' % (sys.argv[0])
        sys.exit()
    simple_deepid(learning_rate=0.01, n_epochs=20, dataset=(sys.argv[1], sys.argv[2]), params_file=sys.argv[3], nkerns=[20,40,60,80], batch_size=500, n_hidden=160, n_out=1595, acti_func=relu)
