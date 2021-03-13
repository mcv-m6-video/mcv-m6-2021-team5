import glob
import os
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

class GaussianBGEstimator:

    """
    Creates GaussianBGEstimator object that reads the images and creates the model
    Params:
        path: path to frames directory
        train_ratio: percentage of frames to use as train set
    """
    def __init__(self, path, train_ratio=0.25):
        self.path = path
        self.train_ratio = train_ratio

    def train(self, color=False):
        """
        This function returns a numpy array of train images
        This assumes that all images have the same size 
        for speed purposes
        """
        # Get image list
        img_list = glob.glob(os.path.join(self.path,'frame_*.png'))

        # Preallocate numpy array for speed
        img_size = np.shape(cv2.imread(img_list[0], cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE))
        w,h = img_size[0:2]
        N = math.floor(self.train_ratio*len(img_list))
        if color:
            self.mean_px = np.zeros((w,h,3))
            self.std_px = np.zeros((w,h,3))
        else:
            self.mean_px = np.zeros((w,h))
            self.std_px = np.zeros((w,h))

        # Two pass method: first compute mean then std
        for i, filename in enumerate(img_list[0:N]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            self.mean_px += img
        self.mean_px /= N

        for i, filename in enumerate(img_list[0:N]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            self.std_px += (img - self.mean_px) * (img - self.mean_px)
        self.std_px /= N-1
        return self.mean_px, self.std_px   
    
    def test(self, indices=None, color=False, alpha=3):
        """
        Test the computed model
        Params:
            indices: indicates the indices of the images to test. If unspecified
                     this will test for all the test images
            color: True: color images, False: grayscale
        Returns:
            ?
        """
        if indices is None:
            img_list = glob.glob(os.path.join(self.path,'frame_*.png'))
            N = len(img_list) - math.floor(self.train_ratio*len(img_list))
            train_N = math.floor(self.train_ratio*len(img_list))
        else: 
            img_list = glob.glob(os.path.join(self.path,'frame_*.png'))
            N = len(img_list) - math.floor(self.train_ratio*len(img_list))
            train_N = math.floor(self.train_ratio*len(img_list))

        for i, filename in enumerate(img_list[train_N:-1]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            foreground = (img-self.mean_px > alpha*self.std_px)
            print(foreground)
            break