import cPickle
import gzip
import os
import sys
import numpy 
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]

class MLP:
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh)
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out)
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                + (self.logRegressionLayer.W ** 2).sum()
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
        data_set_file='face_data/face.gz', batch_size=200, n_hidden=500):
    train_set_x, train_set_y = load_data(data_set_file)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    print '... Build the model'
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)
    classifier = MLP(rng=rng, input=x, n_in=20*20*3,
                     n_hidden=n_hidden, n_out=2)
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr
    
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size: (index+1) * batch_size],
                y: train_set_y[index * batch_size: (index+1) * batch_size]})
    
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index*batch_size: (index+1)*batch_size],
                y: train_set_y[index*batch_size: (index+1)*batch_size]})
    
    print '... Training'
    sample_num = train_set_x.get_value(borrow=True).shape[0]
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for minibach_index in xrange(n_train_batches):
            minibach_cost = train_model(minibach_index)
        test_losses = [test_model(i) for i in xrange(n_train_batches)]
        test_score = numpy.sum(test_losses)
        print 'epoch %i, test_score %f' % (epoch, float(test_score) / sample_num)

def command_prompt(argv):
    if len(argv) != 2:
        print 'Usage: python mlp.py (data_set_file)'
        sys.exit()

if __name__ == '__main__':
    command_prompt(sys.argv)
    test_mlp(learning_rate=0.1, n_epochs=1000, data_set_file = sys.argv[1])

