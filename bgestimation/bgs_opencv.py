import glob
import os
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm
from utils.bb import BB
import pickle as pkl
from utils.plotting import plot_detections
from bgestimation.mask_utils import *

class OpenCVBGEstimators:

    """
    Creates OpenCVBGEstimators object that reads the images and creates the models
    This class is a wrapper for the opencv implemented bg substraction models
    Params:
        img_path: path to input frames directory
    """
    def __init__(self, img_path, train_ratio=0.25):
        self.img_path = img_path
        self.train_ratio = train_ratio

        # Set image list and number of images to use
        self.img_list = sorted(glob.glob(os.path.join(self.img_path,'frame_*.png')))
        self.N_train = math.floor(self.train_ratio*len(self.img_list))
        self.N_test_start = self.N_train
        self.N_test_end = len(self.img_list)
    
    def train(self, models=['MOG2']):
        
        # Create models 
        self.models = {}
        for model in models:
            if model == 'MOG2':
                self.models[model] = cv2.createBackgroundSubtractorMOG2()
            elif model == 'KNN':
                self.models[model] = cv2.createBackgroundSubtractorKNN()
            else:
                print('Invalid model name!')
                return

        # Training
        print('[1] Training opencv models:')
        for filename in tqdm(self.img_list[0:self.N_train]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            for model in models:
                self.models[model].apply(img)


    def test(self, model='MOG2', N_test_start=None, N_test_end=None):
        """
        Tests the opencv background subtraction model for all the test images.
        Returns:
            List of lists of bounding boxes
        """
        # Read params
        if N_test_start is not None:
            self.N_test_start = N_test_start
        if N_test_end is not None:
            self.N_test_end = N_test_end

        # For all the images to test
        detections = []
        print('[1/1] Computing foreground masks for testing frames [' + str(self.N_test_start) + '-' + str(self.N_test_end) + ']:')
        for filename in tqdm(self.img_list[self.N_test_start:self.N_test_end]):
            # Read image
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            frame_name = os.path.split(filename)[1].split(".")[0]
            frame_num = frame_name.split("_")[1]

            # Create a mask with foreground pixels
            foreground_mask = self.models[model].apply(img)

            # Filter
            _, foreground_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
            foreground_mask = denoise_mask(foreground_mask,method=4)

            # Obtain detections
            img = cv2.bitwise_and(img,img,mask=255-foreground_mask)
            contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_dets = []
            j = 1
            for con in contours:
                (x, y, w, h) = cv2.boundingRect(con)
                if w > 20 and h > 10 and w*h > 450 and w < 4*h and h < 4*w: #and w*h < 5E5:
                    frame_dets.append(BB(int(frame_num), None, 'car', x, y, x+w, y+h, 1))
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                    j = j+1
            detections.append(frame_dets)
            
            #cv2.imwrite('./IMAGES/'+model+'/'+frame_name+'.png',img)
        
        return detections

