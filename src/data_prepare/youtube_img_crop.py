#!/usr/bin/env python
# -*- coding:utf8 -*-

import Image
import sys
import os

def crop_img_by_half_center(src_file_path, dest_file_path):
    im = Image.open(src_file_path)
    x_size, y_size = im.size
    start_point_xy = x_size / 4
    end_point_xy   = x_size / 4 + x_size / 2
    box = (start_point_xy, start_point_xy, end_point_xy, end_point_xy)
    new_im = im.crop(box)
    new_new_im = new_im.resize((47,55))
    new_new_im.save(dest_file_path)

def walk_through_the_folder_for_crop(aligned_db_folder, result_folder):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    i = 0
    img_count = 0
    for people_folder in os.listdir(aligned_db_folder):
        src_people_path = aligned_db_folder + people_folder + '/'
        dest_people_path = result_folder + people_folder + '/'
        if not os.path.exists(dest_people_path):
            os.mkdir(dest_people_path)
        for video_folder in os.listdir(src_people_path):
            src_video_path = src_people_path + video_folder + '/'
            dest_video_path = dest_people_path + video_folder + '/'
            if not os.path.exists(dest_video_path):
                os.mkdir(dest_video_path)
            for img_file in os.listdir(src_video_path):
                src_img_path = src_video_path + img_file
                dest_img_path = dest_video_path + img_file
                crop_img_by_half_center(src_img_path, dest_img_path)
            i += 1
            img_count += len(os.listdir(src_video_path))
            sys.stdout.write('\rsub_folder: %d, imgs %d' % (i, img_count) )
            sys.stdout.flush()
    print ''
        
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s aligned_db_folder new_folder' % (sys.argv[0])
        sys.exit()
    aligned_db_folder = sys.argv[1]
    result_folder = sys.argv[2]
    if not aligned_db_folder.endswith('/'):
        aligned_db_folder += '/'
    if not result_folder.endswith('/'):
        result_folder += '/'
    walk_through_the_folder_for_crop(aligned_db_folder, result_folder)
    
