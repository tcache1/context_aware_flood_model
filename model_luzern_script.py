# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 08:26:53 2023

@author: tcache1
"""

"""
- Script for training the urban pluvial flood model 
- Model architecture in file 'model_architecture.py'
- Auxilary functions in file 'auxilary_functions.py'
"""

#%% LIBRARIES AND AUXILARY FUNCTIONS  
import auxilary_functions
import defs
import model_architecture

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import warnings
import multiprocessing
import itertools
from functools import partial
import os
tf.config.run_functions_eagerly(True)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# To make the outputs stable across runs
rnd_seed = 42
rnd_gen = np.random.default_rng(rnd_seed)


#%% DEFINING RAINFALL EVENTS AND PRE-TRAINED MODEL USED FOR TRANSFER LEARNING
# For recall:
# tr0...: [18]
# tr2...: [0, 6, 12, 19]
# tr5...: [1, 7, 13]
# tr10...:[2, 8, 14, 20]
# tr20...:[3, 9, 15]
# tr50...:[4, 10, 21]
# tr100...:[5, 11, 22]
# Can be all except pr = 8 since we don't have the flood map for this event in Luzern
pr_train_index = 0
model_weights = 'model1_weights.h5'


#%% FILE PATHS
dem_path = '...'    
rainfall_file_path = '...'
waterdepth_file_path = '...'


#%% LOADING AND PRE-PROCESSING IMAGE DATA
# Some parameters necessary for the functions 
# Specifying the resolution of the context patches and patch size
res_original = 1
resx2 = 2
resx4 = 4
res_xmax = 4
patch_size = 256
padding = round(patch_size/2*(res_xmax-1))+patch_size

# Loading the DEM
dem = auxilary_functions.load_dem(dem_path)
# New DEM with padding around to enable lower resolution patches extraction
dem = auxilary_functions.pad_dem(dem, padding, res_xmax)

# Mask defining the boundaries of the study area
mask = np.ones_like(dem,dtype=np.float32)
mask[dem==-9999] = 0

# Downscaling the DEM at 2 m and 4 m 
dem_resx2, mask_resx2 = auxilary_functions.lower_res_inputs(dem, mask, 2)
dem_resx4, mask_resx4 = auxilary_functions.lower_res_inputs(dem, mask, 4)

# Extracting the terrain features from the DEM for the 3 different resolutions
def multiple_feature_img(input_dem, input_mask):
    features = auxilary_functions.features(input_dem, input_mask)
    sin = features[0]
    cos = features[1]
    sinks = features[2]
    slope = features[3]
    curv = features[4]
    
    X_img = np.stack((input_dem, sin, cos, sinks, slope, curv), axis=2)
    return X_img

X_img = multiple_feature_img(dem, mask)
X_img_resx2 = multiple_feature_img(dem_resx2, mask_resx2)
X_img_resx4 = multiple_feature_img(dem_resx4, mask_resx4)


#%% LOADING AND SPLITTING RAINFALL EVENTS INTO TRAIN-TEST SETS 
# Loading the rainfall time series and patterns 
X_pr, pattern_names = auxilary_functions.load_rainfall(rainfall_file_path)  

# Value by which the accumulated precip will be 'standardized'
pr_std = np.sum(X_pr[0,:][X_pr[0,:]!=-9999])


#%% LOADING THE 'TARGET' WATERDEPTHS 
# Loading waterdepths corresponding to the rainfall time series  
Y_paths = os.listdir(waterdepth_file_path)
Y_train = []
ij_train = list(itertools.product([pattern_names[pr_train_index]], Y_paths))

for k in range(len(ij_train)): 
    ij = ij_train[k]
    i = ij[0]
    j = ij[1]
    if '_'+i+'_' in j:
        Y_train = defs.f(ij, waterdepth_file_path, padding, dem)


#%% LOADING THE PATCH LOCATIONS, SPLITTING INTO TRAIN-VALIDATION SETS AND IMPLEMENTING PATCH AUGMENTATIONS 
# Loading the patch locations 
# "rb" because we want to read in binary mode
# with open('final_patch_location_Luzern', "rb") as f: 
#     X_coords = pickle.load(f)

# Adding data augmentation
# Each data augmentation combination takes a name which is a number between 1 and 8
# Refer to naming convention in Supplementary Material or 'auxilary_function.py'
# data_augmentation = [1, 2, 3, 4, 5, 6, 7, 8]

# Creating a new array with the patch coordinate pairs in the first 2 columns 
# and the rainfall pattern number in the 3rd column 
# np.repeat: to repeat the coordinate pairs as many times as there are rainfall patterns 
# np.tile: to tile the rainfall patterns as many times as there are coordinate pairs 
# np.column_stack: to have coord pairs and pattern names in the same array 
# train_comb0 = np.column_stack((np.repeat(X_coords, len(data_augmentation), axis=0), 
#                               np.tile(data_augmentation, len(X_coords))))

# Adding precipitation patterns 
# Similarly to the data augmentation technique column, lets add the precipitation 
# event number in the last column
# train_comb = np.column_stack((train_comb0, 
#                               np.tile(pr_train_index, len(train_comb0))))


#%% MODEL'S HYPERPAREMETERS AND INPUT DIMENSIONS 
# Hyperparameters
kernel_size = [4,4]
kernel_init = tf.keras.initializers.glorot_normal
bias_init = 'zeros'
activ_layer = tf.nn.leaky_relu
activ_layer_last = None
pooling_size = [2,2]
dropout_rate = 0.5

# Defining the input dimensions
n_features = X_img.shape[-1]
X_img_shape = [patch_size,patch_size,n_features]
X_img2_shape = X_img_shape
X_img3_shape = X_img_shape
X_rainfall_shape = [None, 1]
pr_std_shape = (1,)

params = {'n_channels':n_features,
         'batch_size':32,
         'patch_size':256}
    

#%% BUILDING, COMPILING THE MODEL AND FREEZING ENCODER LAYERS 
# Option 1: freezing encoder weigths
# encoder_layers = [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 
#                   26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 53, 54, 
#                   55, 56, 71, 73, 75, 77, 78, 85]

# Building and compiling the model 
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# with strategy.scope():
#     multi_model = keras.models.load_model(model_weights, 
#                                           compile=False, options=None)
    
    # Freezing the weights of the encoder layers
#     for i in range(len(encoder_layers)): 
#         index_layer = encoder_layers[i]
#         multi_model.layers[index_layer].trainable = False 
    
#     learning_rate = 0.0001 # default 1e-3
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     multi_model.compile(loss='mse',
#                         optimizer=optimizer, 
#                         metrics=['mae'])

# Option 2: initialize layers (this was the option used to retrain the model)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    multi_model = keras.models.load_model(model_weights, compile=False, options=None)
    
    learning_rate = 0.0001 # default 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    multi_model.compile(loss='mse',
                        optimizer=optimizer, 
                        metrics=['mae'])


#%% DATA GENERATOR 
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_img, X_img_resx2, X_img_resx4, X_pr, pr_std, 
                 waterdepth_img, padding, pr_index, comb_coord_pair, 
                 mask_resx1, mask_resx2, mask_resx4, n_channels, batch_size, patch_size): 
        
        # Initialization of the data generator class 
        self.X_img = X_img
        self.X_img_resx2 = X_img_resx2
        self.X_img_resx4 = X_img_resx4
        self.X_pr = X_pr
        self.pr_std = pr_std
        self.waterdepth_img = waterdepth_img
        self.padding = padding
        self.pr_index = pr_index 
        self.comb_coord_pair = comb_coord_pair
        self.pattern_names = pattern_names 
        self.mask_resx1 = mask_resx1
        self.mask_resx2 = mask_resx2
        self.mask_resx4 = mask_resx4
        self.n_channels = n_channels
        self.batch_size = batch_size 
        self.patch_size = patch_size
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch 
        return int(np.floor(len(self.comb_coord_pair) / self.batch_size))
        
    def __getitem__(self, index):
        # Generates one batch of data 
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find the list of patch coordinates: 
        comb_coord_pair_temp = [self.comb_coord_pair[k] for k in indices]
        
        # Generates data
        X, y = self.__data_generation(comb_coord_pair_temp)
        return X, y
    
    def on_epoch_end(self):
        # Updates indices after each epoch 
        self.indices = np.arange(len(self.comb_coord_pair))
        np.random.shuffle(self.indices)
    
    def __data_generation(self, comb_coord_pair_temp): 
        # Generates data containing batch_size samples 
        X_resx1 = np.empty((self.batch_size, self.patch_size, self.patch_size, self.n_channels))
        X_resx2 = np.empty((self.batch_size, self.patch_size, self.patch_size, self.n_channels))
        X_resx4 = np.empty((self.batch_size, self.patch_size, self.patch_size, self.n_channels))
        X1 = np.empty((self.batch_size, self.X_pr.shape[1]))
        X_std = np.empty((self.batch_size, 1))
        y = np.empty((self.batch_size, self.patch_size, self.patch_size, 1))
        
        for i, j in enumerate(comb_coord_pair_temp): 
            X_resx1[i,] = auxilary_functions.patch_augmentation(
                                    auxilary_functions.minmax_normalization(
                                        np.array(self.mask_resx1[j[0]:j[0]+self.patch_size, j[1]:j[1]+self.patch_size]), 
                                        np.array(self.X_img[j[0]:j[0]+self.patch_size, j[1]:j[1]+self.patch_size])), 
                                    j[-2])
            X_resx2[i,] = auxilary_functions.patch_augmentation(
                                    auxilary_functions.minmax_normalization(
                                        np.array(self.mask_resx2[j[2]:j[2]+self.patch_size, j[3]:j[3]+self.patch_size]), 
                                        np.array(self.X_img_resx2[j[2]:j[2]+self.patch_size, j[3]:j[3]+self.patch_size])), 
                                    j[-2])
            X_resx4[i,] = auxilary_functions.patch_augmentation(
                                    auxilary_functions.minmax_normalization(
                                        np.array(self.mask_resx4[j[4]:j[4]+self.patch_size, j[5]:j[5]+self.patch_size]), 
                                        np.array(self.X_img_resx4[j[4]:j[4]+self.patch_size, j[5]:j[5]+self.patch_size])), 
                                    j[-2])
            X1[i,] = self.X_pr[j[-1]]
            X_std[i,] = np.sum(self.X_pr[j[-1]][self.X_pr[j[-1]]!=-9999])/self.pr_std
            y[i,] = auxilary_functions.patch_augmentation(
                                    self.waterdepth_img[j[0]:j[0]+self.patch_size, 
                                                        j[1]:j[1]+self.patch_size][...,np.newaxis], 
                                    j[-2])

        # Stacking the inputs into one array
        X = [X_resx1, X_resx2, X_resx4, X1, X_std]
        return X, y



#%% TRAINING THE MODEL 
# np.random.shuffle(train_comb)

# ratio = 0.2
# valid_comb = train_comb[:int(len(train_comb)*ratio)]
# train_comb = train_comb[int(len(train_comb)*ratio):]


# Either generate the train/validation/test sets on the fly (in lines 125-173 of this code) 
# or load the patch-rain combinations if they were already generated before. 
# Loading patch-rain combinations that were saved before can be useful for reproducibility 
# purposes. The combinations used to train the model can be found in the Zenodo folder (see readme).
import pickle 
# with open('train_comb_Luzern', 'rb') as f:
#     train_comb = pickle.load(f)
# with open('valid_comb_Luzern', 'rb') as f:
#     valid_comb = pickle.load(f)

# Change the rainfall event to train the model: 
train_comb = train_comb[:,:-1]
train_comb = np.column_stack((train_comb, 
                              np.tile(pr_train_index, len(train_comb))))
valid_comb = valid_comb[:,:-1]
valid_comb = np.column_stack((valid_comb, 
                              np.tile(pr_train_index, len(valid_comb))))


train_generator = DataGenerator(X_img, X_img_resx2, X_img_resx4, X_pr, pr_std, 
                                Y_train, padding, pr_train_index, train_comb, 
                                mask, mask_resx2, mask_resx4, **params)
# We load only one rainfall event - flood map pair, so Y_train = Y_valid
valid_generator = DataGenerator(X_img, X_img_resx2, X_img_resx4, X_pr, pr_std, 
                                Y_train, padding, pr_train_index, valid_comb, 
                                mask, mask_resx2, mask_resx4, **params)

# Defining some callbacks to avoid overfitting the model 
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=1)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('model_weights_luzern_'+pattern_names[pr_train_index]+'.h5',
                                                    save_best_only=True,
                                                    monitor='val_loss')

# ... and finally, training the model! 
history = multi_model.fit(train_generator,
                          epochs=500,
                          validation_data = valid_generator, 
                          callbacks=[early_stopping_cb,
                                      checkpoint_cb]) 

