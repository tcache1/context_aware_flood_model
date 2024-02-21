# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:20:00 2023

@author: tcache1
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.backend as K

def build_cnn(X_img1_shape, X_img2_shape, X_img3_shape, X_rainfall_shape, 
              pr_std_shape, kernel_size, kernel_init, bias_init, activ_layer, 
              activ_layer_last, pooling_size, dropout_rate): 

    
    # Rainfall time series and RNN
    rainfall_input_layer = keras.Input(shape=X_rainfall_shape)
    
    mask = keras.layers.Lambda(lambda rainfall_input_layer: K.not_equal(rainfall_input_layer, -9999))(rainfall_input_layer)
    
    # Normalising by the rainfall accumulated precipitation
    rainfall_std = keras.Input(shape=pr_std_shape)
    
    rainfall_layer = keras.layers.SimpleRNN(20, return_sequences=True)(rainfall_input_layer, mask=mask) 
    rainfall_layer = keras.layers.SimpleRNN(20)(rainfall_layer)
    rainfall_layer = tf.multiply(rainfall_layer, rainfall_std)

    
    
    rainfall_layer = keras.layers.Dense(4096, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer)(rainfall_layer) 
    rainfall_layer = tf.reshape(rainfall_layer,[-1,16,16,16])
    
    
    # VGG16
    f = [32,64,128,256]

    input_layer1 = keras.Input(shape=X_img1_shape)
    input_layer2 = keras.Input(shape=X_img2_shape)
    input_layer3 = keras.Input(shape=X_img3_shape)

    

    conv1_1 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(input_layer1)
    conv1d_1 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv1_1)
    conv1_1 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv1d_1)
    conv2_1 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv1_1)
    conv2d_1 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv2_1)
    conv2_1 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv2d_1)
    conv3_1 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv2_1)
    conv3_1 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_1)
    conv3d_1 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_1)
    conv3_1 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv3d_1)
    conv4_1 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_1)
    conv4_1 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv4_1)
    conv4d_1 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv4_1)
    conv4_1 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv4d_1)
    
    
    
    conv1_2 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(input_layer2)
    conv1_2 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv1_2)
    conv1_2 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv1_2)
    conv2_2 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv1_2)
    conv2_2 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv2_2)
    conv2_2 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv2_2)
    conv3_2 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv2_2)
    conv3_2 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_2)
    conv3_2 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_2)
    conv3_2 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv3_2)
    conv4_2 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_2)
    conv4_2 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv4_2)
    conv4_2 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv4_2)
    conv4_2 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv4_2)
    
    
    
    conv1_3 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(input_layer3)
    conv1_3 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv1_3)
    conv1_3 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv1_3)
    conv2_3 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv1_3)
    conv2_3 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv2_3)
    conv2_3 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv2_3)
    conv3_3 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv2_3)
    conv3_3 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_3)
    conv3_3 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_3)
    conv3_3 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv3_3)
    conv4_3 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv3_3)
    conv4_3 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv4_3)
    conv4_3 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer=kernel_init, bias_initializer=bias_init, activation=activ_layer, padding='same')(conv4_3)
    conv4_3 = keras.layers.MaxPooling2D(pool_size=pooling_size)(conv4_3)
    
    
    # Locality-aware contextual correlation 
    dk = 256
    q1 = tf.keras.layers.Reshape((-1,256))(conv4_1)
    q1 = tf.keras.layers.Permute((2,1))(q1)
    q1 = tf.keras.layers.Dense(dk)(q1)
    k1 = tf.keras.layers.Reshape((-1,256))(conv4_1)
    k1 = tf.keras.layers.Permute((2,1))(k1)
    k1 = tf.keras.layers.Dense(dk)(k1)
    v1 = tf.keras.layers.Reshape((-1,256))(conv4_1)
    v1 = tf.keras.layers.Permute((2,1))(v1)
    v1 = tf.keras.layers.Dense(dk)(v1)
    
    k2 = tf.keras.layers.Reshape((-1,256))(conv4_2)
    k2 = tf.keras.layers.Permute((2,1))(k2)
    k2 = tf.keras.layers.Dense(dk)(k2)
    v2 = tf.keras.layers.Reshape((-1,256))(conv4_2)
    v2 = tf.keras.layers.Permute((2,1))(v2)
    v2 = tf.keras.layers.Dense(dk)(v2)
    
    k3 = tf.keras.layers.Reshape((-1,256))(conv4_3)
    k3 = tf.keras.layers.Permute((2,1))(k3)
    k3 = tf.keras.layers.Dense(dk)(k3)
    v3 = tf.keras.layers.Reshape((-1,256))(conv4_3)
    v3 = tf.keras.layers.Permute((2,1))(v3)
    v3 = tf.keras.layers.Dense(dk)(v3)
    
 
    sim_map1 = tf.matmul(q1, k1, transpose_b=True)/np.sqrt(dk)
    sim_map1 = tf.keras.activations.softmax(sim_map1) 
    out1 = tf.matmul(sim_map1, v1)
    
    sim_map2 = tf.matmul(q1, k2, transpose_b=True)/np.sqrt(dk)
    sim_map2 = tf.keras.activations.softmax(sim_map2) 
    out2 = tf.matmul(sim_map2, v2)
    
    sim_map3 = tf.matmul(q1, k3, transpose_b=True)/np.sqrt(dk)
    sim_map3 = tf.keras.activations.softmax(sim_map3) 
    out3 = tf.matmul(sim_map3, v3)
    
    out = tf.keras.layers.Concatenate()([out1, out2, out3])
    reshape = tf.keras.layers.Reshape((conv4_1.shape[1],conv4_1.shape[2],-1))(out)
    
    
    cat = tf.keras.layers.Concatenate(axis=-1)([reshape, rainfall_layer])
    
    deconv4 = keras.layers.Conv2DTranspose(f[3], pooling_size, strides=(2,2), kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer,  padding='same')(cat)   
    uconv4 = tf.keras.layers.Concatenate()([deconv4,conv4d_1])
    uconv4 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv4)
    uconv4 = keras.layers.Conv2D(f[3], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv4)

    deconv3 = keras.layers.Conv2DTranspose(f[2], pooling_size, strides=(2,2), kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer,  padding='same')(uconv4)   
    uconv3 = tf.keras.layers.Concatenate()([deconv3,conv3d_1])
    uconv3 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv3)
    uconv3 = keras.layers.Conv2D(f[2], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv3)

    deconv2 = keras.layers.Conv2DTranspose(f[1], pooling_size, strides=(2,2), kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer,  padding='same')(uconv3)   
    uconv2 = tf.keras.layers.Concatenate()([deconv2,conv2d_1])
    uconv2 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv2)
    uconv2 = keras.layers.Conv2D(f[1], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv2)

    deconv1 = keras.layers.Conv2DTranspose(f[0], pooling_size, strides=(2,2), kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer,  padding='same')(uconv2)   
    uconv1 = tf.keras.layers.Concatenate()([deconv1,conv1d_1])
    uconv1 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv1)
    uconv1 = keras.layers.Conv2D(f[0], kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv1)

    output_layer =  keras.layers.Conv2D(1, kernel_size, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activ_layer, padding='same')(uconv1)

    model = keras.Model(inputs=[input_layer1, input_layer2, input_layer3, rainfall_input_layer, rainfall_std], outputs=[output_layer])
    
    return model

        
    