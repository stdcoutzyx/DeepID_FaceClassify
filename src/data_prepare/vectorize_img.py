#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import sys
import random
import numpy as np
import Image

def read_csv_file(csv_file):
    path_and_labels = []
    f = open(csv_file, 'rb')
    for line in f:
        line = line.strip('\r\n')
        path, label = line.split(',')
        label = int(label)
        path_and_labels.append((path, label))
    f.close()
    random.shuffle(path_and_labels)
    return path_and_labels

def vectorize_imgs(path_and_labels, image_size):
    image_vector_len = np.prod(image_size)
    
    arrs   = []
    labels = [] 
    i = 0
    for path_and_label in path_and_labels:
        path, label = path_and_label
        img = Image.open(path)
        arr_img = np.asarray(img, dtype='float64')
        arr_img = arr_img.transpose(2,0,1).reshape((image_vector_len, ))
        
        labels.append(label)
        arrs.append(arr_img)

        i += 1
        if i % 100 == 0:
            sys.stdout.write('\rdone: ' + str(i))
            sys.stdout.flush()
    print ''
    arrs = np.asarray(arrs, dtype='float64')
    labels = np.asarray(labels, dtype='int32')
    return (arrs, labels)

def cPickle_output(vars, file_name):
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def output_data(vector_vars, vector_folder, batch_size=1000):
    if not vector_folder.endswith('/'):
        vector_folder += '/'
    if not os.path.exists(vector_folder):
        os.mkdir(vector_folder)
    x, y = vector_vars
    n_batch = len(x) / batch_size
    for i in range(n_batch):
        file_name = vector_folder + str(i) + '.pkl'
        batch_x = x[ i*batch_size: (i+1)*batch_size]
        batch_y = y[ i*batch_size: (i+1)*batch_size]
        cPickle_output((batch_x, batch_y), file_name)
    if n_batch * batch_size < len(x):
        batch_x = x[n_batch*batch_size: ]
        batch_y = y[n_batch*batch_size: ]
        file_name = vector_folder + str(n_batch) + '.pkl'
        cPickle_output((batch_x, batch_y), file_name)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: python %s test_set_file train_set_file test_vector_folder train_vector_folder' % (sys.argv[0])
        sys.exit()
    test_set_file  = sys.argv[1]
    train_set_file = sys.argv[2]
    test_vector_folder  = sys.argv[3]
    train_vector_folder = sys.argv[4]
    
    test_path_and_labels  = read_csv_file(test_set_file)
    train_path_and_labels = read_csv_file(train_set_file)

    print 'test  img num: %d' % (len(test_path_and_labels))
    print 'train img num: %d' % (len(train_path_and_labels))

    img_size = (3, 55, 47)  # channel, height, width
    test_vec  = vectorize_imgs(test_path_and_labels, img_size)
    train_vec = vectorize_imgs(train_path_and_labels, img_size)

    output_data(test_vec,  test_vector_folder)
    output_data(train_vec, train_vector_folder)

