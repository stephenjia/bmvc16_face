from skimage import io
from skimage import color
from skimage import transform
import numpy as np

import os
from os import listdir
from os.path import isfile, join

import cPickle as pickle
import json

img_size = (60,60,3) 

dataset_path = '/esat/ruchba/xjia/image_generation/multi_pie/face_aligned_60_color/'

Recording_id = '01'

pose_list = {'080':0, '130':1, '140':2, '051':3, '050':4, '041':5, '190':6} # [-45d to 45d]

epsilon = 1e-6

#import pdb; pdb.set_trace()

# image name is of the format 'subject001_session01_recording_id01_pose080_illumination00.png'
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

pose_size = 7

dirs = sorted(listdir(dataset_path))
# note that in the folder there are images out of 45d

cnt = 0
for img_file in dirs: 
    print img_file
    img_id = img_file[:-4]
    img_id_terms = img_id.split('_')
    
    if not img_id_terms[3] in pose_list:   continue    
    
    img_color = io.imread(join(dataset_path, img_file))
    img_mean = np.mean(img_color.reshape((60*60,3)), axis=0)
    img_std = np.std(img_color.reshape((60*60,3)), axis=0)
    img = np.divide(img_color-img_mean[np.newaxis,np.newaxis,:], img_std[np.newaxis, np.newaxis, :])
    
    img_blob={}
    img_blob['subject'] = int(img_id_terms[0])    
    pose_idx = pose_list[img_id_terms[3]]
    img_blob['pose_idx'] = pose_idx
    img_blob['illumination_idx'] = int(img_id_terms[4])
    img_blob['Recording_id'] = '01'
    img_blob['session'] = '01'
    img_blob['img_id'] = cnt
    pose_code = np.zeros((1,pose_size))
    pose_code[0, pose_idx] = 1
    if int(img_id_terms[0])<=100:            
        train_set.append(img_blob)
        train_img.append(img.reshape((1,img_size[0],img_size[1], img_size[2])))
        train_code.append(pose_code)
        train_mean_list.append(img_mean)
        train_var_list.append(img_std**2)
    else:
        test_set.append(img_blob)
        test_img.append(img.reshape((1,img_size[0],img_size[1], img_size[2])))
        test_code.append(pose_code)
        test_mean_list.append(img_mean)
        test_var_list.append(img_std**2)
        
    cnt += 1
               
train_img = np.concatenate(train_img, axis=0)
train_code = np.concatenate(train_code, axis=0)
test_img = np.concatenate(test_img, axis=0)
test_code = np.concatenate(test_code, axis=0)
fod = open('/esat/ruchba/xjia/image_generation/multi_pie/dataset_60_color_normalized_aligned.pkl', 'wb')
pickle.dump([train_set, train_img, train_code, train_mean_list, train_var_list, test_set, test_img, test_code, test_mean_list, test_var_list], fod)  
fod.close()   
