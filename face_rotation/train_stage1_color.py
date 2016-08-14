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
from model_stage1_color import Model
from lasagne.updates import rmsprop, adam, sgd, apply_momentum
from lasagne.layers import get_all_params, get_all_layers, get_output, count_params
from lasagne.objectives import squared_error, binary_crossentropy

from config_stage1_color import *

lasagne.random.set_rng(np.random.RandomState(1234))

""" load model """
def load_model(layers, filepath):
    # load checkpoint
    checkpoint = pickle.load(open(filepath, 'rb'))
    model_values = checkpoint['model_values'] 
    lasagne.layers.set_all_param_values(layers, model_values) # overwrite the values of model parameters
    return layers

""" save model """
def save_model(filepath, layers, options, epoch, history_train):
    # model.layers
    model_values = lasagne.layers.get_all_param_values(layers, trainable=True)
    # save model
    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['model_values'] = model_values # values of model parameters
    checkpoint['layers'] = layers
    checkpoint['history_train'] = history_train
    checkpoint['options'] = options        
            
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print 'saved checkpoint in %s' % (filepath, )
    except Exception, e: # todo be more clever here
        print 'tried to write checkpoint into %s but got error: ' % (filepath, )
        print e       


# -------- main function ------------------
if options['debug_flag'] is True:
    import pdb; pdb.set_trace()
host = socket.gethostname() # get computer hostname
options['host'] = host
options['time_start'] = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

# prepare data            
print('Prepare data...')  
fid = open(os.path.join(options['dataset_root'],'dataset_32_prepared_aligned.pkl'),'rb')
train_pair, test_pair = pickle.load(fid)
fid.close()
fid = open(os.path.join(options['dataset_root'],'dataset_60_color_normalized_aligned.pkl'), 'rb')
train_set, train_img, train_code, train_mean_list, train_var_list, test_set, test_img, test_code, test_mean_list, test_var_list = pickle.load(fid)
fid.close()
_, w, h, c = train_img.shape
options['img_size'] = (w,h,c) 

# ---------- build model and compile ---------------
loss_fun = squared_error
# input
img_batch = T.tensor4() # (batch_size, channel, npx, npx)
pose_code = T.matrix() 
img_batch_target = T.tensor4() 
print('Build model...')
model = Model(options)
print('Compile ...')
net, l_pose_reshape = model.build_model(img_batch, pose_code)
layers = get_all_layers(net)
all_params = get_all_params(layers, trainable = True)
# compute loss 
generation = lasagne.layers.get_output(net) 
generation = generation.dimshuffle([0,2,3,1])
# mean squared error
train_loss = lasagne.objectives.squared_error(generation.reshape((generation.shape[0],-1)), img_batch_target.reshape((img_batch_target.shape[0],-1)))    
train_loss = train_loss.sum(axis=1)
train_loss = train_loss.mean()     
# update
lrn_rate = T.cast(theano.shared(options['learning_rate']), 'floatX') # dynamic learning rate
optimizer = sgd
updates_sgd = optimizer(train_loss, all_params, learning_rate=lrn_rate) 
updates = apply_momentum(updates_sgd, all_params, momentum=0.95)
# train
_train = theano.function([img_batch, pose_code, img_batch_target], train_loss, updates=updates, allow_input_downcast=True)

# ------------ training ----------------
split = 'train'
print("Train...")
if options['start_epoch']==0:
    start_epoch = 0
else:
    model.load_model(options['init_model_from'])
    start_epoch = options['start_epoch'] 

nb_epoch = options['max_epochs']
batch_size = options['nbatch']
verbose = options.get('verbose',1)
shuffle = options.get('shuffle', True)    
img_size = options['img_size']

index_array = np.arange(len(train_pair))
print 'Start training ...'
history_train = {} 
for epoch in range(start_epoch, start_epoch+nb_epoch):
    start_time = time.time()
    if options['shuffle']:
        np.random.shuffle(index_array)   
    
    nb_batch = int(np.ceil(len(index_array)/float(batch_size)))
    history_batch = []
    for batch_index in range(0, nb_batch):
        batch_start = batch_index*batch_size
        batch_end = min(len(index_array), (batch_index+1)*batch_size)
        batch_ids = index_array[batch_start:batch_end]
        
        batch_train_pair = [train_pair[id] for id in batch_ids]                
        batch_train_x_ids = [tup[0] for tup in batch_train_pair]
        batch_train_y_ids = [tup[1] for tup in batch_train_pair]
        
        x_train_img = np.reshape(train_img[batch_train_x_ids], (len(batch_ids),img_size[0],img_size[1],img_size[2]))
        y_train_code = train_code[batch_train_y_ids]
        y_train_img = np.reshape(train_img[batch_train_y_ids], (len(batch_ids),img_size[0],img_size[1],img_size[2]))        
        
        # train
        loss_train = _train(x_train_img, y_train_code, y_train_img)
        
        if options['verbose']:
            is_last_batch = (batch_index == nb_batch - 1)
            if not is_last_batch:
                print("Epoch {} of {}, batch {} of {}, took {:.3f}s".format(epoch + 1, nb_epoch, batch_index+1, nb_batch, time.time() - start_time))
                print("  training loss:\t{:.6f}".format(loss_train.item()))
                history_batch.append(loss_train)

            else:                          
                print '\n' 
                print time.time() - start_time                        
                history_batch.append(loss_train)
                history_train[str(epoch)] = np.mean(history_batch)
                print("  training loss:\t{:.6f}".format(history_train[str(epoch)].item()))
                history_batch = []
                if (epoch+1) % options['period'] == 0:
                    loss = history_train[str(epoch)]
                    filename = 'model_checkpoint_%s_%s_%s_%s_epoch%d_train_%.3f.p' % (options['dataset'], options['time_start'], options['host'], options['fappend'], epoch, loss)
                    filepath = os.path.join(options['checkpoint_output_directory'], filename)
                    save_model(filepath, layers, options, epoch, history_train)

