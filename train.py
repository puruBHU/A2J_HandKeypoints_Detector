# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:48:14 2020

@author: Purnendu Mishra
"""

# Standard library import
import argparse
import numpy as np
import pandas as pd
from   pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import Huber

# Local library import
from CustomDataloader import DataAugmentor
from Model import A2J

#%%
parser = argparse.ArgumentParser()

parser.add_argument('-b','--batch_size', default=1, type=int, 
                    help='Batch size to be used for training')

parser.add_argument('-e','--epochs', default = 100, type=int, 
                    help='Defines the number of epochs the model will be trained')

args = parser.parse_args()

#%%
target_size    = (176,) * 2

batch_size     = args.batch_size 
epochs         = args.epochs

#%%

# root         = Path('../../Dataset')
root         = Path.home()/'Documents'/'DATASETS'
train_file   = Path.cwd()/'onehand10k_train_files.txt'
val_file     = Path.cwd()/'onehand10k_test_files.txt'


#%%
train_data   = DataAugmentor(rescale          = 1/255.0,
                             horizontal_flip  = True,
                             vertical_flip    = True)

train_loader = train_data.flow_from_directory(root        = root,
                                              data_file   = train_file, 
                                              target_size = target_size,
                                              mode        = 'train',
                                              batch_size  = batch_size,
                                              shuffle     = True)


val_data    = DataAugmentor(rescale = 1/255.0)

val_loader  = val_data.flow_from_directory(root        = root,
                                           data_file   = val_file, 
                                           target_size = target_size,
                                           mode        = 'test',
                                           batch_size  = batch_size,
                                           shuffle     = False
                                           )

#%%
huber  = Huber(delta=1.0)

model  = A2J(input_shape = target_size + (3,), keys = 21)
opt    = Adam(lr = 0.00035, beta_1 = 0.5)  
# opt      = SGD(lr = initial_lr, momentum=0.9, nesterov=True, decay=1e-6)

model.compile(optimizer    = opt,
              loss         = huber,
              metrics      = ['mae']
              )

# weights = 'A2J_OneHand_SGD_109_0.0206.hdf5'
# model.load_weights(weights, by_name = True, skip_mismatch=True)
#%%
save_path = Path.cwd()/'checkpoints'

if not save_path.exists():
  save_path.mkdir(parents = True)
  

checkpoints = ModelCheckpoint('{}/A2J_OneHand_Cropped_'.format(save_path) + '{epoch:03d}_{val_loss:0.4f}.hdf5',
                              monitor        = 'val_loss',
                              mode           = 'auto',
                              verbose        = 1,
                              save_best_only = True, 
                              save_freq      = 'epoch'
                              )

records     = CSVLogger('logs/A2J_OneHand_Cropped_training.log')

callbacks   = [checkpoints, records]
#%%
print('Started training the model ...')
model.fit(x                   = train_loader,
          validation_data     = val_loader,
          epochs              = epochs,
          steps_per_epoch     = len(train_loader),
          callbacks           = callbacks,
          use_multiprocessing = False,
          workers             = 4,
          verbose             = 2,
          initial_epoch       = 0
          )

model.save('A2J_OneHand__Cropped_Final_weights.hdf5')
