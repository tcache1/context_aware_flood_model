# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:12:44 2022

@author: tcache1
"""

# STEP 1: 
    # Load the data 
    # Generate features from the DEM 
    # Min-max normalize the input images

import pandas as pd 
import numpy as np
import richdem as rd
from PIL import Image
import tensorflow as tf


# Loading the DEM 
def load_dem(dem_file_path):
    print("Loading the DEM:", dem_file_path)
    dem = None
    # Reading the ascii DEM and removing the header (ncols, nrows, xllcorner, yllcorner, cellsize and nodata value)
    dem = pd.read_csv(dem_file_path, header=None, skiprows=6, delimiter=" ", skipinitialspace = True, dtype=np.float64)
    # DEM to rdarray as required by RichDEM
    dem = rd.rdarray(dem, no_data=-9999)
    return dem


# Function to add padding around the DEM
# This is necessary to be able to extract the wider context patches
def pad_dem(dem, padding, res_xmax):
    # Add padding such that the lower resolution window can capture the whole 
    # patch if the patch location is at a border 
    dem_padding = np.ones(np.add(dem.shape,2*padding))*(-9999)
    
    # Make the dimensions of the image even numbers to avoid issues when resizing 
    # to lower resolution
    column_or_row = [1, 0]
    for j in range(0,2): 
        n_new = res_xmax-dem_padding.shape[j]%res_xmax
        if n_new != 0:
            if j == 0: 
                c = np.ones((n_new, dem_padding.shape[column_or_row[j]]))*(-9999)
            elif j == 1:
                c = np.ones((dem_padding.shape[column_or_row[j]], n_new))*(-9999)
            dem_padding = np.concatenate((dem_padding, c), axis=j)
    
    dem_padding[padding:padding+dem.shape[0],padding:padding+dem.shape[1]] = dem
    dem = rd.rdarray(dem_padding, no_data=-9999)
    return dem


# Downscaling the DEM image 
def lower_res_inputs(feature, mask, XresxX):
    # XresxX is the ratio of the size of the cell of lower resolution images and original resolution
    fe_resxX = Image.fromarray(feature)
    (width, height) = (fe_resxX.width // XresxX, fe_resxX.height // XresxX)
    fe_resxX = fe_resxX.resize((width, height), resample=4)
    fe_resxX = np.array(fe_resxX)
    
    mask_resxX = Image.fromarray(mask)
    (width, height) = (mask_resxX.width // XresxX, mask_resxX.height // XresxX)
    mask_resxX = mask_resxX.resize((width, height), resample=4)
    mask_resxX = np.array(mask_resxX)
    mask_resxX[(mask_resxX!=1)&(mask_resxX!=0)]=0
    
    # Remove cells at boundary
    fe_resxX[mask_resxX==0]=-9999
    return fe_resxX, mask_resxX


# Feature extraction: https://richdem.readthedocs.io/en/latest/python_api.html
def features(dem, mask):
    dem = rd.rdarray(dem, no_data=-9999)
    # Filling the DEM
    dem_fill = rd.FillDepressions(dem)
    
    # Aspect: characterizes flow direction on terrains
    aspect = rd.TerrainAttribute(dem_fill, attrib='aspect')
    sin = np.sin(np.pi*aspect/180)
    cos = np.cos(np.pi*aspect/180)
    sin[mask==0] = 0
    cos[mask==0] = 0
    
    # Sinks: depth of depressions in the terrain 
    sinks = dem_fill-dem
    sinks[mask==0] = 0
    
    # Slope: in percentage 
    slope = rd.TerrainAttribute(dem_fill, attrib='slope_percentage')
    
    # Curvature: mean curvature
    Zy, Zx  = np.gradient(dem_fill)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    curv = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx
    curv = -curv/(2*(Zx**2 + Zy**2 + 1)**(1.5))
    curv[mask==0] = 0

    return sin, cos, sinks, slope, curv



# Loading the rainfall time series and their name (used to load corresponding waterdepths)
def load_rainfall(rainfall_file_path):
    data = pd.read_csv(rainfall_file_path, header=None, index_col=0, delimiter="\t", skipinitialspace=True)
    values = data.values
    index = data.index
    return values, index


# Finds the patch coordinates corresponding to the lower resolution images 
def multi_scale_patches(patch_coord, patch_size, res_original, res_new):
    x = res_new/res_original
    patch_coord_multi_scale = patch_coord - patch_size/2*(x-1)
    patch_coord_multi_scale = patch_coord_multi_scale/x
    
    return np.around(patch_coord_multi_scale)


# Combinations of flips and rotation to augment the data
# Refer to the naming convention for the functions number 
def patch_augmentation(patch, aug):
    if aug ==1:
        new_patch = patch
        
    if aug==2:
        new_patch = np.rot90(patch)
        # new_patch = np.array(tf.image.rot90(patch))
        # new_patch = np.array([[patch[j][i] for j in range(len(patch))] for i in range(len(patch[0])-1,-1,-1)])
    if aug==3:
        new_patch = np.flipud(np.fliplr(patch))
        # new_patch = np.array(tf.image.flip_up_down(tf.image.flip_left_right(patch)))
        # new_patch = np.transpose(np.array([[patch[j][i] for j in range(len(patch)-1,-1,-1)] for i in range(len(patch[0])-1,-1,-1)]),axes=(1,0,2))
    if aug==4: 
        new_patch = np.rot90(np.rot90(np.rot90(patch)))
        # new_patch = np.array(tf.image.rot90(tf.image.rot90(tf.image.rot90(patch))))
        # new_patch = np.array([[patch[j][i] for j in range(len(patch)-1,-1,-1)] for i in range(len(patch[0]))])
    if aug==5:
        # new_patch = np.array(tf.image.rot90(tf.image.rot90(tf.image.rot90(tf.image.flip_up_down(patch)))))
        new_patch = np.transpose(patch,axes=(1,0,2))
    if aug==6:
        new_patch = np.fliplr(patch)
        # new_patch = np.array(tf.image.flip_left_right(patch))
        # new_patch = np.transpose(np.array([[patch[j][i] for j in range(len(patch))] for i in range(len(patch[0])-1,-1,-1)]),axes=(1,0,2))
    if aug==7:
        new_patch = np.rot90(np.rot90(np.rot90(np.fliplr(patch))))
        # new_patch = np.array(tf.image.rot90(tf.image.rot90(tf.image.rot90(tf.image.flip_left_right(patch)))))
        # new_patch = np.array([[patch[j][i] for j in range(len(patch)-1,-1,-1)] for i in range(len(patch[0])-1,-1,-1)])
    if aug==8:
        new_patch = np.flipud(patch)
        # new_patch = np.array(tf.image.flip_up_down(patch))
        # new_patch = np.transpose(np.array([[patch[j][i] for j in range(len(patch)-1,-1,-1)] for i in range(len(patch[0]))]),axes=(1,0,2))  
    return new_patch 


# Min-max normalization [-1, 1]
def minmax_normalization(mask, feature): 
    mask_indices = mask==1
    
    # If there is only one feature that is being fed to the model (e.g., only DEM):
    if len(feature.shape)==2:
        l_max = feature[mask_indices].max()
        l_min = feature[mask_indices].min()
        if l_max==l_min: 
            normalized_feature = feature
        else: 
            normalized_feature = (feature-l_min)/(l_max-l_min)*2-1
            normalized_feature = normalized_feature*mask
            
    # If there is more than just one feature image (e.g., DEM and slope):
    elif len(feature.shape)==3:
        normalized_feature = np.zeros(feature.shape)
        for i in range(0,feature.shape[-1]):
            fe = feature[:,:,i]
            l_max = fe[mask_indices].max()
            l_min = fe[mask_indices].min()
            if l_max==l_min: 
                normalized_feature[:,:,i] = fe
            else:
                normalized_feature[:,:,i] = (fe-l_min)/(l_max-l_min)*2-1
                normalized_feature[:,:,i] = normalized_feature[:,:,i]*mask 
    return normalized_feature


