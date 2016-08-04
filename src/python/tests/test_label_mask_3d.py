# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:31:23 2016

@author: saadiaiftikhar
"""

import unittest
from nose.tools import *
from label_mask_3d import *
import numpy as np
import cv2
import os
from skimage.filters import threshold_otsu
from remove_small_objects_3d import *
import mahotas as mahotas
import requests
from urlparse import urlparse

minarea = 100

INPUT_FILES = ("p00_D03_y000_x000_t000_c002_z000.tif")

OUTPUT_FILES = ("p00_D03_y000_x000_t000_c002_z000_Output_Image.tif")

IMAGES_URL = "http://testdata.tissuemaps.org/storage/test_data/"

TEMP_PATH = "/Users/saadiaiftikhar/Desktop/Test_Data/"

try:
    os.stat(TEMP_PATH)
except:
    os.mkdir(TEMP_PATH)  


class test_label_mask_3d(unittest.TestCase):
    
    def test_label_3d_actual_data(self): 

        username = 'storage'
        password = 'o62ReU98h9Yb'
        input_file_url = IMAGES_URL + INPUT_FILES
        input_filename = os.path.basename(urlparse(input_file_url).path)

        temp_input_file = TEMP_PATH + input_filename 
        r = requests.get(input_file_url, auth=(username,password))
        if r.status_code == 200:
            with open(temp_input_file, 'wb') as out:
                for bits in r.iter_content():
                    out.write(bits)
        
        output_file_url = IMAGES_URL + OUTPUT_FILES
        output_filename = os.path.basename(urlparse(output_file_url).path)

        temp_output_file = TEMP_PATH + output_filename       
        r = requests.get(output_file_url, auth=(username,password))
        if r.status_code == 200:
            with open(temp_output_file, 'wb') as out:
                for bits in r.iter_content():
                    out.write(bits)
                
        intesity_images = cv2.imread(temp_input_file, cv2.IMREAD_UNCHANGED) 
        
        im_test_output = cv2.imread(temp_output_file, cv2.IMREAD_UNCHANGED) 
        
        gaussed_intensity_image = mahotas.gaussian_filter(intesity_images,8)
         
        thresh1 = threshold_otsu(gaussed_intensity_image)
         
        thresholded_orig_image = gaussed_intensity_image > thresh1
        
        thresholded_orig_image,foo = remove_small_objects_3d(
                                            thresholded_orig_image,minarea)
        
        label_image = label_mask_3d(thresholded_orig_image)
        
        L = np.uint16(label_image['label_image'])

        assert (len(np.unique(L))-1 == len(np.unique(im_test_output))-1
                                        ), "labeling the images works fine"

        assert (len(np.unique(im_test_output))-1 == 136
                                        ), "labeling the images works fine"
        

    def test_label_3d_fake_data(self): 
        
        A = np.zeros((128,128), np.int)
        L1 = label_mask_3d(A)
        L = L1['label_image']
        assert not L.max()
    
        A[2:5, 2:5] = 34
        A[10:50, 10:50] = 34
        L1 = label_mask_3d(A)
        L = L1['label_image']
        assert L.max() == 2
        assert np.sum(L > 0) == (40*40 + 3*3)
        assert np.all( (L > 0) == (A > 0) )
        assert set(L.ravel()) == set([0,1,2])
    
    def test_all_ones(self):
        
        A = np.ones((32,32))
        L1 = label_mask_3d(A)
        L = L1['label_image']
        assert np.all(L == 1)

    def test_random(self):
        
        np.random.seed(33)
        A = np.random.rand(128,128) > .8
        L1 = label_mask_3d(A)
        L = L1['label_image']
        assert len(set(L.ravel())) == (len(np.unique(L))-1+1)
        assert L.max() == len(np.unique(L))-1
    
    
if __name__ == '__main__':
    unittest.main()


        
        
        