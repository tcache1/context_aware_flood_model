# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:08:54 2023

@author: tcache1
"""

"""
- Function used to download the target waterdepth files in parallel
"""


import pandas as pd 
import numpy as np 
import gzip

def f(ij, waterdepth_file_path, padding, dem): 
    i = ij[0]
    j = ij[1]
    waterdepth = []
    if '_'+i+'_' in j: 
        if ((i=='tr10-2') & ('luzern' in j)):
            pass
        else:
            if '.gz' in j:
                with gzip.open(str(waterdepth_file_path)+'/'+str(j)) as wd_pattern_file_gzip:
                    waterdepth = pd.read_csv(wd_pattern_file_gzip, 
                                        header=None, 
                                        skiprows=6,
                                        delimiter=" ", 
                                        skipinitialspace =True, 
                                        dtype=np.float32).values
                # Some waterdepth files have NaN end column (e.g. Luzern)
                if np.any(np.isnan(waterdepth[:,-1])): 
                    waterdepth = waterdepth[:,:-1]
            
            elif '.gz' not in j: 
                wd_pattern_file_asc = str(waterdepth_file_path)+'/'+str(j)
                waterdepth = pd.read_csv(wd_pattern_file_asc, 
                                        header=None, 
                                        skiprows=6,
                                        delimiter=" ", 
                                        skipinitialspace =True, 
                                        dtype=np.float32).values
                
                # Some waterdepth files have NaN end column (e.g. Luzern)
                if np.any(np.isnan(waterdepth[:,-1])): 
                    waterdepth = waterdepth[:,:-1]
    
    
    if type(waterdepth)==np.ndarray:
        # Add padding such that the waterdepth image has the same dimension 
        # (i.e., padding) as the terrain images. 
        waterdepth_pad = np.zeros(dem.shape)
        waterdepth_pad[padding:padding+waterdepth.shape[0],
                       padding:padding+waterdepth.shape[1]] = waterdepth
        return waterdepth_pad
    else:
        return waterdepth
    
    
