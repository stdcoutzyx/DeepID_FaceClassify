import cPickle
import gzip
import os
import sys
import time
import numpy

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
                value=numpy.zeros( (n_in, n_out), dtype=theano.config.floatX), 
                name='W', 
                borrow=True)
        self.b = theano.shared(
                value=numpy.zeros( (n_out,), dtype=theano.config.floatX),
                name='b',
                borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should hava the same shape as self.y_pred',
                    ('y', target.type, 'y_pred', self.y_pred,type))
        if y.dtype.startswith('int'):
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(data_set_file):
    f = gzip.open(data_set_file, 'rb')
    train_set = cPickle.load(f)
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
    
    return shared_dataset(train_set)

def sgd_face(learning_rate=0.13, n_epochs=300, batch_size=300, data_set_file='face_data/face.gz'):
    train_set_x, train_set_y = load_data(data_set_file)
    n_train_batches= train_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... Build the model'
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in = 20*20*3, n_out=2)
    cost = classifier.negative_log_likelihood(y)
    
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index*batch_size: (index+1)*batch_size],
                y: train_set_y[index*batch_size: (index+1)*batch_size]})
    
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index*batch_size: (index+1)*batch_size],
                y: train_set_y[index*batch_size: (index+1)*batch_size]})
    
    print '... Training the model'

    sample_num = train_set_x.get_value(borrow=True).shape[0]
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
        test_losses = [test_model(i) for i in xrange(n_train_batches)]
        test_score = numpy.sum(test_losses)
        print 'epoch %i, test_score %f' % (epoch, float(test_score) / sample_num)

if __name__ == '__main__':
    sgd_face(learning_rate=0.13, n_epochs=2000, data_set_file = sys.argv[1])

