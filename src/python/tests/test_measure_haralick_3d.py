# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:31:23 2016

@author: saadiaiftikhar
"""

import unittest
from nose.tools import *
from measure_haralick_3d import *
import numpy as np
import cv2
from skimage.measure import label
import os
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
    

class test_measure_haralick_3d(unittest.TestCase):
    
    def test_measure_haralick_3d_actual_data(self): 
        
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
                
        intensity_image = cv2.imread(temp_input_file, cv2.IMREAD_UNCHANGED) 
        
        im_test_output = cv2.imread(temp_output_file, cv2.IMREAD_UNCHANGED) 
        label_image = cv2.imread(im_test_output, cv2.IMREAD_UNCHANGED) 
        label_image = label(label_image > 0)
       
        measurements = measure_haralick_3d(label_image, intensity_image, plot=False)
        measurements_haralick = measurements['measurements']

        intensity_image = np.zeros((20, 20), dtype=np.uint16)
        intensity_image[13:15, 13:17] = 230
        
        label_image = np.zeros((20, 20), dtype=np.double)
        label_image[13:15, 13:17] = 1
        label_image = label(label_image > 0)
        
        measurements1 = measure_haralick_3d(label_image, intensity_image, plot=False)
        measurements_haralick1 = measurements1['measurements']

        assert (measurements_haralick.shape[1] == measurements_haralick1.shape[1]
                                    ), "measurements of the images are correct"
        

    def test_measure_haralick_3d_fake_data(self): 
        
        intensity_image = np.zeros((20, 20), dtype=np.uint16)
        intensity_image[13:15, 13:17] = 230
        
        label_image = np.zeros((20, 20), dtype=np.double)
        label_image[13:15, 13:17] = 1
        label_image = label(label_image >0)
        
        measurements1 = measure_haralick_3d(label_image, intensity_image, plot=False)
        measurements_haralick1 = measurements1['measurements']
        
        
        intensity_image2 = np.zeros((20, 20), dtype=np.uint16)
        intensity_image2[11:13, 11:13] = 150
        
        label_image2 = np.zeros((20, 20), dtype=np.double)
        label_image2[11:13, 11:13] = 1
        label_image2 = label(label_image2 >0)
        
        measurements2 = measure_haralick_3d(label_image2,intensity_image2, plot=False)
        measurements_haralick2 = measurements2['measurements']

#        assert_almost_equal(measurements_hu1, measurements_hu2, decimal=1)
        
        assert (measurements_haralick1.shape[1] == measurements_haralick2.shape[1]
                                    ), "measurements of the images are correct"
        
    
    
if __name__ == '__main__':
    unittest.main()


        
        
        