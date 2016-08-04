# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:46:26 2016

@author: saadiaiftikhar
"""
import unittest
from nose.tools import *
from label_mask_3d import *
from expand_objects_3d import *
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

OUTPUT_FILES = ("p00_D03_y000_x000_t000_c002_z000_Expanded_Image.tif")

IMAGES_URL = "http://testdata.tissuemaps.org/storage/test_data/"

TEMP_PATH = "/Users/saadiaiftikhar/Desktop/Test_Data/"

try:
    os.stat(TEMP_PATH)
except:
    os.mkdir(TEMP_PATH)  


class test_expand_objects_3d(unittest.TestCase):
    
    def test_expand_objects_3d_actual_data(self): 
        
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
        
        label_image_result = label_mask_3d(thresholded_orig_image)
        label_image = np.uint16(label_image_result['label_image'])
        
        n1 = 5
        expanded = expand_objects_3d(label_image, n1, plot=False)
        expanded_label_image = np.int32(expanded['expanded_image'])
        

        assert(len(np.unique(expanded_label_image))-1 == 
            len(np.unique(im_test_output))-1), "labeling the images works fine"
        

    def test_expand_objects_3d_fake_data(self):
        
        label_image = np.zeros((10, 10), int)
        label_image[4, 4] = 1
        
        expected_image = np.zeros((10, 10), int)
        expected_image[np.array([4, 3, 4, 5, 4], int), 
                       np.array([3, 4, 4, 4, 5], int)] = 1
        
        n1 = 5
        expanded = expand_objects_3d(label_image, n1, plot=False)
        expanded_label_image = np.int32(expanded['expanded_image'])
        
        assert(len(np.unique(expanded_label_image))-1 == 
            len(np.unique(expected_image))-1), "labeling the images works fine"

if __name__ == '__main__':
    unittest.main()
   