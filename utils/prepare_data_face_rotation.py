# prepare training pairs and test pairs
# for each sample, there should be 
# img_ori, code_ori, img_new, code_new

import numpy as np
import cPickle as pickle

img_file = '/esat/ruchba/xjia/image_generation/multi_pie/dataset_32_grayscale_normalized_aligned.pkl'
img_size = (32,32)
pose_size = 7
illum_size = 20

#import pdb; pdb.set_trace()
    
full_size = illum_size+pose_size

fid = open(img_file, 'rb')
train_set, train_img, train_code, train_mean_list, train_var_list, test_set, test_img, test_code, test_mean_list, test_var_list = pickle.load(fid)
fid.close()

## training set
train_pair = []
for x_sample in train_set: 
    x_img_id = x_sample['img_id'] 
    print x_img_id
    
    sel_subject = x_sample['subject']
    sel_pose_list = range(pose_size)
    sel_pose_list.remove(x_sample['pose_idx'])
    
    # same id and same illumination
    y_samples = filter(lambda d: d['subject']==sel_subject and d['pose_idx'] in sel_pose_list and d['illumination_idx']==x_sample['illumination_idx'], train_set)
    
    pair_list_tmp = [(x_img_id, y_sample['img_id']) for y_sample in y_samples ]
    train_pair = train_pair + pair_list_tmp
    

## test set
test_pair = []
for x_sample in test_set:        
    x_img_id = x_sample['img_id']-len(train_set) 
    print x_img_id        
    
    sel_subject = x_sample['subject']
    sel_pose_list = range(pose_size)
    sel_pose_list.remove(x_sample['pose_idx'])
    
    # same id and same pose
    y_samples = filter(lambda d: d['subject']==sel_subject and d['pose_idx'] in sel_pose_list and d['illumination_idx']==x_sample['illumination_idx'], test_set)
    
    pair_list_tmp = [(x_img_id, y_sample['img_id']-len(train_set) ) for y_sample in y_samples ]
    test_pair = test_pair + pair_list_tmp           
        
     
fod = open('/esat/ruchba/xjia/image_generation/multi_pie/dataset_32_prepared_aligned.pkl','wb')
pickle.dump([train_pair, test_pair], fod)  
fod.close()   
