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

illum_size = 20 # illumination

dirs = sorted(listdir(dataset_path))
# note that in the folder there are images out of 45d

cnt = 0
for img_file in dirs: 
    print img_file
    img_id = img_file[:-4]
    img_id_terms = img_id.split('_')
    
    if not img_id_terms[3] in pose_list:   continue    
    
    # illumination code
    illum_idx = int(img_id_terms[4])    
    illum_code = np.zeros((1,illum_size))
    illum_code[0, illum_idx] = 1
    if int(img_id_terms[0])<=100:            
        train_code.append(illum_code)
    else:
        test_code.append(illum_code)        
    cnt += 1
               
train_code = np.concatenate(train_code, axis=0)
test_code = np.concatenate(test_code, axis=0)
fod = open('/esat/ruchba/xjia/image_generation/multi_pie/dataset_60_color_normalized_aligned_illum.pkl', 'wb')
pickle.dump([train_code, test_code], fod)  
fod.close()   
