#!/usr/bin/env python
# -*-coding:utf8-*-

import os, sys
import numpy
import Image
import random
import gzip
import cPickle
import pickle

def data_shuffle(origin_root):
    map_label = {}
    file_path_and_labels = []
    for folder_name in os.listdir(origin_root):
        folder_path = origin_root + folder_name
        label_index = len(map_label)
        map_label[folder_name] = label_index
        
        for file_name in os.listdir(folder_path):
            file_path = folder_path + '/' + file_name
            file_path_and_labels.append((file_path, label_index))
    random.shuffle(file_path_and_labels)
    return file_path_and_labels, map_label

def train_valid(file_path_and_labels, rate=0.1):
    file_num = len(file_path_and_labels)
    valid_num = int(file_num * rate)
    train_num = file_num - valid_num
    train_files = file_path_and_labels[0:train_num]
    valid_files = file_path_and_labels[train_num:]
    return train_files, valid_files

def make_array(file_path_and_labels, image_size):
    image_vector_len = numpy.prod(image_size)
    arr = []
    labels = []

    num = len(file_path_and_labels)
    i = 0
    for file_path_and_label in file_path_and_labels:
        file_path, label = file_path_and_label
        labels.append(label)
        img = Image.open(file_path)
        arr_img = numpy.asarray(img, dtype='float64')
        arr_img = arr_img.swapaxes(0,2).swapaxes(1,2)
        vec_img = arr_img.reshape((image_vector_len, ))
        arr.append(vec_img)
        i += 1
        if i % 100 == 0:
            sys.stdout.write('\rdone: ' + str(i) + '/' + str(num))
            sys.stdout.flush()
    print ''

    arr = numpy.asarray(arr, dtype='float64')
    labels = numpy.asarray(labels, dtype='int32')
    return (arr, labels)

def dump2file(dest_file, arr_data):
    f = open(dest_file, 'wb')
    pickle.dump(arr_data, f)
    f.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s (input_folder) (output_file)' % sys.argv[0]
        sys.exit()

    origin_root = sys.argv[1]
    dest_file   = sys.argv[2]

    image_size = (47,55,3) # width, height, channel
    rate = 0.11
    if not origin_root.endswith('/'):
        origin_root += '/'
    
    file_path_and_labels, map_label = data_shuffle(origin_root)
    train_files, valid_files = train_valid(file_path_and_labels, rate)
    print len(map_label)
    print len(file_path_and_labels)
    
    print 'make array ...'
    train_arr = make_array(train_files, image_size)
    print 'train_arr shape:', train_arr[0].shape, train_arr[1].shape
    valid_arr = make_array(valid_files, image_size)
    print 'valid_arr shape:', valid_arr[0].shape, valid_arr[1].shape
    arr_data = [train_arr, valid_arr]

    print 'dump files ...'
    dump2file(dest_file, arr_data)
