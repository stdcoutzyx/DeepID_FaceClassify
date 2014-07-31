#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import sys
from PIL import Image
import numpy as np
import gzip
import cPickle

class ImagePredeal:
    def __init__(self, face_src_path, noface_src_path, out_file):
        self.face_src_path = face_src_path
        self.noface_src_path = noface_src_path
        self.out_file = out_file
    
    def get_file_names(self):
        face_files   = get_list_files(self.face_src_path)
        noface_files = get_list_files(self.noface_src_path)
        face_files   = [self.face_src_path + filename for filename in face_files if filename.endswith('.bmp')]
        noface_files = [self.noface_src_path + filename for filename in noface_files if filename.endswith('.bmp')]
        print ''
        print 'face files num  :\t', len(face_files)
        print 'noface files num:\t', len(noface_files)
        return face_files, noface_files
    
    def read_images(self, face_files, noface_files):
        imgs = np.zeros((1,1200))
        labels = [0]
        for filename in face_files:
            img = self.read_image(filename)
            if img == None:
                continue
            imgs = np.row_stack((imgs, img))
            labels.append(1)
        for filename in noface_files:
            img = self.read_image(filename)
            if img == None:
                continue
            imgs = np.row_stack((imgs, img))
            labels.append(0)

        imgs = imgs[1:]
        labels = np.array(labels[1:])
        print imgs.shape
        print labels.shape
        return (imgs, labels)
    
    def trainset_2_file(self, train_sets):
        p = cPickle.dumps(train_sets)
        f_out = gzip.open(self.out_file, 'wb')
        f_out.write(p)
        f_out.close()

    def read_image(self, filename):
        img = Image.open(open(filename))
        img = np.asarray(img, dtype='float64') / 256
        n = 1
        for d in img.shape:
            n *= d
        if n != 1200:
            print filename, img.shape
            return None
        img = img.reshape((1,1200))
        return img

def get_list_files(path):
    file_list = os.listdir(path)
    return file_list

def command_prompt(argv):
    length_of_params = len(argv)
    if length_of_params != 4:
        print 'Usage: python image_predeal.py (face_src_path) (noface_src_path) (out_file)'
        sys.exit()

if __name__ == '__main__':
    command_prompt(sys.argv)

    face_src_path = sys.argv[1]
    noface_src_path = sys.argv[2]
    out_file = sys.argv[3]
    
    if not face_src_path.endswith('/'):
        face_src_path += '/'
    if not noface_src_path.endswith('/'):
        noface_src_path += '/'

    print 'face_src_path  :\t', face_src_path
    print 'noface_src_path:\t', noface_src_path
    print 'out_file       :\t', out_file
   
    ip = ImagePredeal(face_src_path, noface_src_path, out_file)
    face_files, noface_files = ip.get_file_names()
    train_sets = ip.read_images(face_files, noface_files)
    ip.trainset_2_file(train_sets)
