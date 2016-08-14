# generation
# export THEANO_FLAGS="device=gpu0, floatX=float32" optimizer=None

import numpy as np
import theano
import theano.tensor as T

import sys
import os
import argparse
import cPickle as pickle
import socket
import datetime
import time

import lasagne 
from model_stage1_color import Model as Model_stage1
from model_stage2_color import Model as Model_stage2

from lasagne.layers import get_all_params, get_all_layers, get_output
import skimage.io
#import matplotlib.pyplot as plt

lasagne.random.set_rng(np.random.RandomState(1234))

import pdb; pdb.set_trace()

# ----------- prepare data -------------------------
pose_list = {0:'080', 1:'130', 2:'140', 3:'051', 4:'050', 5:'041', 6:'190'} # [-45d to 45d]
print('Prepare data...')  
fid = open(os.path.join('/esat/ruchba/xjia/image_generation/multi_pie/','dataset_32_prepared_aligned.pkl'),'rb')
train_pair, test_pair = pickle.load(fid)
fid.close()
fid = open(os.path.join('/esat/ruchba/xjia/image_generation/multi_pie/','dataset_60_color_normalized_aligned.pkl'), 'rb')
train_set, train_img, train_code, train_mean_list, train_var_list, test_set, test_img, test_code, test_mean_list, test_var_list = pickle.load(fid)
fid.close()
_, w, h, c = train_img.shape

# ---------- build model and compile ---------------
# input
img_batch = T.tensor4()  # (batch_size, rgb, npx, npx)
pose_code = T.matrix() 

print('Build model stage1...')
checkpoint_1 = pickle.load(open('/esat/tiger/xjia/image_generation/multipie/model_checkpoint_multi_pie_16-04-13-10-10_andromeda.esat.kuleuven.be_rotate_aligned_60_color_onebranch_epoch199_train_810.648.p', 'rb')) # the model trained at the first stage
model_stage1 = Model_stage1(checkpoint_1['options'])
net_1, l_pose_reshape = model_stage1.build_model(img_batch, pose_code)
layers_1 = get_all_layers(net_1)
generation_1 = lasagne.layers.get_output(net_1)
generation_1 = generation_1.dimshuffle([0,2,3,1]) # from [batch_size, channel, npx, npx] to [batch_size, npx, npx, channel]
_generate_1 = theano.function([img_batch, pose_code], generation_1, allow_input_downcast=True) 
lasagne.layers.set_all_param_values(layers_1, checkpoint_1['model_values'], trainable=True)

print('Build model stage2...')
img_batch_gen = T.tensor4()
img_batch_target = T.tensor4()
checkpoint_2 = pickle.load(open('/esat/tiger/xjia/image_generation/multipie/model_checkpoint_multi_pie_16-04-19-10-13_andromeda.esat.kuleuven.be_rotate_aligned_60_recurrent_epoch199_train_592.584.p', 'rb'))
model_stage2 = Model_stage2(checkpoint_2['options'])
net_2 = model_stage2.build_model(img_batch, img_batch_gen)
layers_2 = get_all_layers(net_2)
all_params_2 = get_all_params(layers_2, trainable = True)
generation_2 = lasagne.layers.get_output(net_2)       
generation_2 = generation_2.dimshuffle([0,2,3,1])
_generate_2 = theano.function([img_batch, img_batch_gen], generation_2, allow_input_downcast=True) 
lasagne.layers.set_all_param_values(layers_2, checkpoint_2['model_values'], trainable=True)

# ----------- generation -------------
result_path = '/esat/ruchba/xjia/image_generation/multi_pie/DeconvNet_aligned_60_new/img_gen_demo_color/' 
if not os.path.isdir(result_path):
    os.makedirs(result_path) 
    
idx_all = np.arange(len(test_pair))
np.random.seed(1000)
np.random.shuffle(idx_all)
idx_selected = idx_all[0:10000]
idx_selected = np.sort(idx_selected)

for (cnt,sub) in enumerate(idx_selected):
    print(cnt)
    x_idx = test_pair[sub][0]
    y_idx = test_pair[sub][1]
    test_sample_x_img = test_img[x_idx]
    test_sample_y_code = test_code[y_idx]
    test_sample_y_img = test_img[y_idx]
    
    test_predict_tmp = _generate_1(test_sample_x_img.reshape((1,w,h,c)), test_sample_y_code.reshape((1,7)))    
    test_predict = _generate_2(test_sample_x_img.reshape((1,w,h,c)), test_predict_tmp)
    generated_img_tmp = np.reshape(test_predict,(w,h,c))
    
    # source image
    test_std_src = np.sqrt(test_var_list[x_idx])
    test_mean_src = test_mean_list[x_idx]
    source_img = test_sample_x_img*test_std_src[np.newaxis,np.newaxis,:]+test_mean_src[np.newaxis,np.newaxis,:]
    source_img = source_img * (source_img>0)        

    # generated image
    test_std = np.sqrt(test_var_list[y_idx])
    test_mean = test_mean_list[y_idx]
    generated_img = generated_img_tmp *test_std[np.newaxis,np.newaxis,:]+test_mean[np.newaxis,np.newaxis,:] # x_idx, y_idx
    generated_img = generated_img * (generated_img > 0)
    
    # target image
    target = test_set[y_idx]
    target_img = test_sample_y_img *test_std[np.newaxis,np.newaxis,:]+test_mean[np.newaxis,np.newaxis,:]
    target_img = target_img * (target_img>0)   
    
    # concatenation
    concat_img = np.concatenate([source_img, target_img, generated_img],axis=1) 
    """
    # matplotlib
    plt.figure(1)
    plt.imshow(concat_img, cmap=plt.get_cmap('gray'))
    plt.show()
    """
    # save image
    filename = '%d_%03d_01_01_%s_%02d.png' % (sub,target['subject'], pose_list[target['pose_idx']], target['illumination_idx'])
    skimage.io.imsave(result_path + filename, np.uint8(concat_img))

