# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:02:58 2023

@author: tcache1
"""

import numpy as np 
import matplotlib.pyplot as plt
import auxilary_functions


city = 'Singapore_2m'


#%% SOME FUNCTIONS 

# Verifies that the 2 requirements are fulfilled:
# 1. that 2 patches are far enough to each other so to limit overlap
# 2. that at least 10% of the patch covers the study area 
def checks(new_point, points, r_threshold, mask, min_area_prop, patch_size):
    for point in points:
        dist = np.sqrt(np.sum(np.square(new_point-point)))
        x = new_point[0]
        y = new_point[1]
        #dist_min = dist.min()
        if dist < r_threshold or np.count_nonzero(mask[x:x+patch_size,y:y+patch_size]) < min_area_prop*patch_size*patch_size:
            return False
    print(new_point)
    return True

# Randomly generating patch locations and verifying that the requirements are fulfilled
def RandX(N, r_threshold, mask, min_area_prop, patch_size, padding):
    # The minimum value of patch coordinate is: padding-patch_size 
    # The results would not change if xy_min = 0 since if x or y = 0 then less of 
    # 10% of the patch would cover the study area. 
    # Adding the xy_min contraint simply speeds up the process as the code does 
    # not need to test conditions for patches for which we already know lay outside the study area. 
    xy_min = padding-patch_size 
    # xy_max!= xy_min since the coordinate define the upper left corner of the patch 
    xy_max = padding
    scope = np.arange(xy_min,max(mask.shape)-xy_max,1)
    points = [np.random.choice(scope, 2)]
    while len(points) < N:
        new_point = np.random.choice(scope, 2)
        if checks(new_point, points, r_threshold, mask, min_area_prop, patch_size):
            points.append(new_point)
    return points

# Function that can be used to find the minimum distance between the patches
# Not used when generating patches but can be used to check requirement 1
def dist_points(points):
    dists = []
    for point_eval in points:
        dist = 10000
        for point in points:
            if (point_eval!=point).any():
                dist_new = np.sqrt(np.sum(np.square(point_eval-point)))
                if dist_new < dist: 
                    dist = dist_new
        dists.append(dist)
    return points, dists



#%% PARAMETERS AND LOADING THE DATA 

patch_size = 256
min_area_prop = 0.1
min_dist_prop = 0.23
r_threshold = min_dist_prop*patch_size
if city == 'Zurich':
    N = 1250
elif city == 'Luzern':
    N = 620
elif city == 'Singapore':
    N = 1500
elif city == 'Singapore_2m':
    N = 450
# We add 1 as the first point might be generated outside of the study 
# area and will be removed from the list later
N = N+1 


if city == 'Zurich':
    dem_path = r'...'  
if city == 'Luzern': 
    dem_path = r'...'
elif city == 'Singapore':
    dem_path = r'...'
elif city == 'Singapore_2m':
    dem_path = r'...'


dem = auxilary_functions.load_dem(dem_path)
if 'Singapore' in city: 
    if np.any(np.isnan(dem[:,-1])): 
        dem = dem[:,:-1]
    

res_xmax = 4
resx2 = 2
resx4 = 4
res_original = 1

padding = round(patch_size/2*(res_xmax-1))+patch_size
dem = auxilary_functions.pad_dem(dem, padding, res_xmax)
dem[np.isnan(dem)]=-9999

mask = np.zeros((max(dem.shape),max(dem.shape)),dtype=np.float32)-9999
mask[0:dem.shape[0],0:dem.shape[1]] = dem
mask[mask==-9999]=0
        

#%% GENERATING THE PATCH LOCATIONS
coord = RandX(N, r_threshold, mask, min_area_prop, patch_size, padding)
# Remove the first coordinates as these have been randomly picked and might be 
# located outside of the training area 
X_coord_train = np.array(coord[1:])

# Patch coordinates for lower resolution images 
# Resolution x2
X_coord_train_resx2 = auxilary_functions.multi_scale_patches(X_coord_train, patch_size, res_original, resx2)
# Resolution x4
X_coord_train_resx4 = auxilary_functions.multi_scale_patches(X_coord_train, patch_size, res_original, resx4)

X_coord_train = np.column_stack((X_coord_train, X_coord_train_resx2, X_coord_train_resx4)).astype(int)


#%% VISUALISATION OF PATCH LOCATIONS 

# 1: Visualising the locations of the patches upper left corner
patch_loc = dem*0
for i,j in enumerate(X_coord_train):
   patch_loc[j[0]:j[0]+20,j[1]:j[1]+20] = 3

# 2: Visualising the density of the patches
patch_loc = dem*0
for i,j in enumerate(X_coord_train):
     patch_loc[j[0]:j[0]+patch_size,j[1]:j[1]+patch_size] = patch_loc[j[0]:j[0]+patch_size,j[1]:j[1]+patch_size]+1

        
plt.rcParams['font.size'] = 12
fig, ax1 = plt.subplots(figsize=(18,6), nrows=1, ncols=1)

y1 = ax1.imshow(patch_loc, cmap='gist_heat_r', vmin=0)
fig.colorbar(y1, ax=ax1, fraction=0.046, pad=0.04)


#%% SAVING THE PATCH LOCATIONS 

import pickle 
if city == 'Zurich':
    with open('final_patch_location_Zurich', 'wb') as f:
        pickle.dump(X_coord_train, f)
elif city == 'Luzern':
    with open('final_patch_location_Luzern', 'wb') as f:
        pickle.dump(X_coord_train, f)
elif city == 'Singapore':
    with open('final_patch_location_Sgp', 'wb') as f:
        pickle.dump(X_coord_train, f)
elif city == 'Singapore_2m':
    with open('final_patch_location_Sgp_2m', 'wb') as f:
        pickle.dump(X_coord_train, f)

