# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:17:13 2023

@author: tcache1
"""

"""
- Script for generating flood maps using the urban pluvial flood model 
- Model architecture in file 'model.py'
- Auxilary functions in file 'auxilary_functions.py'
"""

#%% LIBRARIES AND AUXILARY FUNCTIONS  
import auxilary_functions
import model_architecture
import supplementary_functions

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import pickle
import warnings
import multiprocessing
import itertools
from functools import partial
import os
from matplotlib import pyplot as plt, cm
tf.config.run_functions_eagerly(True)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# To make the outputs stable across runs
rnd_seed = 42
rnd_gen = np.random.default_rng(rnd_seed)


#%% DEFINING THE CITY AND RAINFALL EVENT FOR WHICH THE PREDICTIONS MUST BE MADE 
# For recall, the validation rainfall events are: [9, 13, 22] = [tr20-2, tr5-3, tr100-4=tr100-2]
# and the test rainfall events are: [0, 5, 8, 16] = [tr2, tr100, tr10-2, tr50-3]

# City and Rainfall Event for which the flood map will be generated:
city = 'sgp'
pr = 'tr100'

# City and Rainfall Event used for training the model:
city_training = 'sgp'
pr_training = 'tr2'

model_weights = 'model_weights_'+city_training+'_'+pr_training+'.h5'


#%% FILE PATHS
# Rainfall file path:
rainfall_file_path = 'C:\\Users\\tcache1\\OneDrive - Université de Lausanne\\Flood_simulations_PhD2\\1D_flood_emulation\\Guo 2020 code+data\\simulationdata\\744\\rain_pattern_str.txt'
# Terrain (city) file path:
if city=='zurich':
    dem_path = 'C:\\Users\\tcache1\\OneDrive - Université de Lausanne\\Flood_simulations_PhD2\\1D_flood_emulation\\Guo 2020 code+data\\simulationdata\\744\\dem\\744_dem\\744_dem_asc.asc'    
if city=='luzern':
    dem_path = 'C:\\Users\\tcache1\\OneDrive - Université de Lausanne\\Flood_simulations_PhD2\\1D_flood_emulation\\Guo 2020 code+data\\simulationdata\\luzern\\dem\\luzern.asc'    
if city=='sgp':
    dem_path = r'C:\Users\tcache1\OneDrive - Université de Lausanne\Flood_simulations_PhD2\1D_flood_emulation\Jovan\sgorchard_2m.asc'
# The DEMs of Guo et al. 2022 ar defined by numbers 
if city.isdigit()==True:
    dem_path = 'C:\\Users\\tcache1\\OneDrive - Université de Lausanne\\Flood_simulations_PhD2\\1D_flood_emulation\\data1D\\dem\\'+city+'_dem.asc'    


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
if np.any(np.isnan(dem[:,-1])): 
    dem = dem[:,:-1]
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


#%% LOADING RAINFALL EVENTS 
# Rainfall time series and patterns 
X_pr, pattern_names = auxilary_functions.load_rainfall(rainfall_file_path)  

# Finding the index corresponding to the rainfall event for which we are generating 
# the flood map.
pr_index = np.where(pattern_names==pr)

pr_std = np.sum(X_pr[0,:][X_pr[0,:]!=-9999])


#%% GENERATING PATCH LOCATION ON REGULAR GRID 
min_area_prop = 0.1
# step at which patches are generated
step = 128
# Generating the patches location for the high resolution image
X_coord_resx1 = supplementary_functions.patch_locations(dem, mask, padding, patch_size, step, min_area_prop)
# Converting the patch locations for the lower resolution images (contextual information patches)
X_coord_resx2 = auxilary_functions.multi_scale_patches(X_coord_resx1, patch_size, res_original, resx2)
X_coord_resx4 = auxilary_functions.multi_scale_patches(X_coord_resx1, patch_size, res_original, resx4)
X_coord = np.column_stack((X_coord_resx1, X_coord_resx2, X_coord_resx4)).astype(int)
# Stacking together the patch locations and the rainfall event index 
X_coord = np.column_stack(( X_coord , np.squeeze(np.tile(pr_index, len(X_coord)))))


#%% MODEL'S HYPERPARAMETERS INPUT DIMENSIONS 
# Defining the input dimensions
n_features = X_img.shape[-1]
X_img_shape = [patch_size,patch_size,n_features]
X_img2_shape = X_img_shape
X_img3_shape = X_img_shape
X_rainfall_shape = [None, 1]
pr_std_shape = (1,)

params = {'n_channels':n_features, 
          'batch_size':1,
          'patch_size':256} 

# Optimizer and learning rate 
learning_rate = 0.0001 # default 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


#%% LOADING THE MODEL ARCHITECTURE AND ITS WEIGHTS, AND COMPILITNG THE MODEL 
# Loading the weights of the best model trained 
model = keras.models.load_model(model_weights, compile=False, options=None,
                                custom_objects={'K':K})
model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae'])


#%% DATA GENERATOR
# The data generator is identical to the one for training the model, except that 
# it does not generate the output image Y (which was generated for the supervised
# training of the model)
class DataGenerator(keras.utils.Sequence):
    def __init__(self, X_img, X_img_resx2, X_img_resx4, X_pr, pr_std, padding, 
                 comb_coord_pair, mask_resx1, mask_resx2, mask_resx4, n_channels, 
                 batch_size, patch_size): 
        # Initialization of the data generator class 
        self.X_img = X_img
        self.X_img_resx2 = X_img_resx2
        self.X_img_resx4 = X_img_resx4
        self.X_pr = X_pr
        self.pr_std = pr_std
        self.padding = padding
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
        
        # Generate data: 
        X = self.__data_generation(comb_coord_pair_temp)
        return X
    
    def on_epoch_end(self):
        # Updates indices after each epoch 
        self.indices = np.arange(len(self.comb_coord_pair))
    
    def __data_generation(self, comb_coord_pair_temp): 
        # Generates data containing batch_size samples 
        X_resx1 = np.empty((self.batch_size, self.patch_size, self.patch_size, self.n_channels))
        X_resx2 = np.empty((self.batch_size, self.patch_size, self.patch_size, self.n_channels))
        X_resx4 = np.empty((self.batch_size, self.patch_size, self.patch_size, self.n_channels))
        X1 = np.empty((self.batch_size, self.X_pr.shape[1]))
        X_std = np.empty((self.batch_size, 1))

        # Scaling at patch scale:
        for i, j in enumerate(comb_coord_pair_temp): 
            mask_low_res = np.array(self.mask_resx4[j[4]:j[4]+self.patch_size, j[5]:j[5]+self.patch_size])
            feature_low_res = np.array(self.X_img_resx4[j[4]:j[4]+self.patch_size, j[5]:j[5]+self.patch_size])
            
            X_resx1[i,] = auxilary_functions.minmax_normalization(
                                    np.array(self.mask_resx1[j[0]:j[0]+self.patch_size, j[1]:j[1]+self.patch_size]), 
                                    np.array(self.X_img[j[0]:j[0]+self.patch_size, j[1]:j[1]+self.patch_size])) 
            X_resx2[i,] = auxilary_functions.minmax_normalization(
                                    np.array(self.mask_resx2[j[2]:j[2]+self.patch_size, j[3]:j[3]+self.patch_size]), 
                                    np.array(self.X_img_resx2[j[2]:j[2]+self.patch_size, j[3]:j[3]+self.patch_size]))    
            X_resx4[i,] = auxilary_functions.minmax_normalization(
                                    np.array(self.mask_resx4[j[4]:j[4]+self.patch_size, j[5]:j[5]+self.patch_size]), 
                                    np.array(self.X_img_resx4[j[4]:j[4]+self.patch_size, j[5]:j[5]+self.patch_size]))  
            X1[i,] = self.X_pr[j[-1]]
            X_std[i,] = np.sum(self.X_pr[j[-1]][self.X_pr[j[-1]]!=-9999])/self.pr_std
            
        # Stacking the inputs into one array
        X = [X_resx1, X_resx2, X_resx4, X1, X_std]
        return X


#%% GENERATING THE FLOOD MAP
data_generator = DataGenerator(X_img, X_img_resx2, X_img_resx4, X_pr, pr_std, padding, X_coord, mask, mask_resx2, mask_resx4, **params)

# Generating patch flood maps and combining the patches into the city flood map 
# Patches are aggregated by averaging the generated flood maps 
# Initializing the array
y = np.zeros((X_img.shape[0], X_img.shape[1]))
S = np.zeros((X_img.shape[0], X_img.shape[1]))
# Saving the predictions (non-parallelized version):  
for i,j in enumerate(X_coord):
    y[j[0]:j[0]+patch_size,j[1]:j[1]+patch_size] += np.squeeze(model.predict(data_generator[i]))
    S[j[0]:j[0]+patch_size,j[1]:j[1]+patch_size] += np.ones((patch_size, patch_size))
    print(i)
y[S>0] = y[S>0]/S[S>0]


#%% VISULAZING THE FLOOD MAP
plt.rcParams['font.size'] = 12
fig, ax1 = plt.subplots(figsize=(18,6), nrows=1, ncols=1)

cmap = cm.terrain
y1 = ax1.imshow(y[650+padding:1040+padding,1500+padding:1880+padding], vmin=0, vmax=y[mask==1].max(), cmap=cmap)
cbar = fig.colorbar(y1, ax=ax1, fraction=0.046, pad=0.04)
                    # ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
# cbar.ax.set_yticklabels(['0.00 m','0.05 m','0.10 m','0.50 m','1.00 m','3.00 m','7.00 m'])


# Saving the results
# import pickle 
# if city_training=='zurich':
#     with open('ZURICH\\wd_'+city+'_'+pr, 'wb') as f:
#         pickle.dump(y, f)
# elif city_training=='luzern':
#     with open('LUZERN\\'+pr_training+'\\wd_'+city+'_'+pr, 'wb') as f:
#         pickle.dump(y, f)
# elif city_training=='sgp':
#     with open('SGP\\'+pr_training+'\\wd_'+city+'_'+pr, 'wb') as f:
#         pickle.dump(y, f)


import winsound
winsound.Beep(500,1500)


