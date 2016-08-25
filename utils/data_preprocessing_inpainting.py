from skimage import io
from skimage import color
from skimage import transform
import numpy as np

import os
from os import listdir
from os.path import isfile, join

import cPickle as pickle
import json

img_size = (60,60,3)  # color iamge (60,60,3)
pose_list = {'080':0, '130':1, '140':2, '051':3, '050':4, '041':5, '190':6} # [-45d to 45d]
pose_size = 7

dataset_path = '/users/visics/aghodrat/codes/eyescream/multipie/datasets/detector/per_image_dpm/'

epsilon = 1e-6

"""occluded image"""
fod = open('/esat/ruchba/xjia/image_generation/multi_pie/dataset_60_grayscale_normalized_syn_color.pkl', 'wb')
train_set = []
train_img = []
train_code = []
train_mean_list = []
train_var_list = []

test_set = []
test_img = []
test_code = []
test_mean_list = []
test_var_list = []

dirs = sorted(listdir(os.path.join(dataset_path, 'images_attr_gen3')))

for img_file in dirs[1:]: # the first file is .
    print img_file
    img_id = img_file[:-4]
    img_id_terms = img_id.split('_')
    
    if not img_id_terms[3] in pose_list:   continue 
    
    img_color = io.imread(join(dataset_path, 'images_attr_gen3', img_file))                
    img_mean = np.mean(img_color.reshape((60*60,3)), axis=0)
    img_std = np.std(img_color.reshape((60*60,3)), axis=0)
    img = np.divide(img_color-img_mean[np.newaxis,np.newaxis,:], img_std[np.newaxis, np.newaxis, :])  
    
    pose_code = np.zeros((1,pose_size))
    pose_code[0,-1] = 1
    
    if int(img_id_terms[0])<=100:            
        train_set.append(img_id)
        train_img.append(img.reshape((1,img_size[0],img_size[1], img_size[2])))
        train_code.append(pose_code)
        train_mean_list.append(img_mean)
        train_var_list.append(img_std**2)
    else:
        test_set.append(img_id)
        test_img.append(img.reshape((1,img_size[0],img_size[1], img_size[2])))
        test_code.append(pose_code)
        test_mean_list.append(img_mean)
        test_var_list.append(img_std**2)
                
train_img = np.concatenate(train_img, axis=0)
train_code = np.concatenate(train_code, axis=0)
test_img = np.concatenate(test_img, axis=0)
test_code = np.concatenate(test_code, axis=0)

pickle.dump([train_set, train_img, train_code, train_mean_list, train_var_list, test_set, test_img, test_code,test_mean_list, test_var_list], fod)  
fod.close()            
                                   
