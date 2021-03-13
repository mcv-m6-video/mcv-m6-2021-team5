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

class GaussianBGEstimator:

    """
    Creates GaussianBGEstimator object that reads the images and creates the model
    Params:
        img_path: path to input frames directory
        mask_path: path to output masks directory
        train_ratio: percentage of frames to use as train set
    """
    def __init__(self, img_path, mask_path, train_ratio=0.25):
        self.img_path = img_path
        self.mask_path = mask_path

        # Create mask path
        if os.path.exists(mask_path):
            shutil.rmtree(mask_path)
        os.makedirs(mask_path)

        self.train_ratio = train_ratio

    def load_pretrained(self, filename):
        with open(filename, 'rb') as f:
            self.mean_px, self.std_px = pkl.load(f)

    def save_trained(self, filename):
        with open(filename,'wb') as f:
            pkl.dump([self.mean_px, self.std_px], f)

    def train(self, color=False):
        """
        This function returns a numpy array of train images
        This assumes that all images have the same size 
        for speed purposes
        """

        print('Training estimator:')

        # Get image list and number of images to use
        img_list = sorted(glob.glob(os.path.join(self.img_path,'frame_*.png')))
        N_train = math.floor(self.train_ratio*len(img_list))

        # Get image size with first image
        img_size = np.shape(cv2.imread(img_list[0], cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE))
        w,h = img_size[0:2]

        # Preallocate numpy array for speed
        if color:
            self.mean_px = np.zeros((w,h,3))
            self.std_px = np.zeros((w,h,3))
        else:
            self.mean_px = np.zeros((w,h))
            self.std_px = np.zeros((w,h))

        # Two pass method: first compute mean then std
        print('[1/2] Computing mean for training frames (' + str(np.shape(img_list[0:N_train])[0]) + '/' + str(np.shape(img_list)[0]) + '):')
        for filename in tqdm(img_list[0:N_train]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            self.mean_px += img
        self.mean_px /= N_train

        print('[2/2] Computing std for training frames (' + str(np.shape(img_list[0:N_train])[0]) + '/' + str(np.shape(img_list)[0]) + '):')
        for filename in tqdm(img_list[0:N_train]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            self.std_px += (img - self.mean_px) * (img - self.mean_px)
        print(filename)
        self.std_px = np.sqrt(self.std_px/(N_train-1))

        return self.mean_px, self.std_px   
    
    def test(self, indices=None, color=False, alpha=1):
        """
        Test the computed model
        Params:
            indices: indicates the indices of the images to test. If unspecified
                     this will test for all the test images
            color: True: color images, False: grayscale
        Returns:
            List of lists of BBs [[BB,BB...],...]
        """
        print('Testing estimator:')

        # Get image list and number of images to use
        if indices is None:
            img_list = sorted(glob.glob(os.path.join(self.img_path,'frame_*.png')))
            N_train = math.floor(self.train_ratio*len(img_list))
            N_test = len(img_list) - N_train
        else: 
            img_list = sorted(glob.glob(os.path.join(self.img_path,'frame_*.png')))
            N_train = math.floor(self.train_ratio*len(img_list))
            N_test = len(img_list) - N_train

        # For all the images to test
        detections = []
        print('[1/1] Computing foreground masks for testing frames (' + str(np.shape(img_list[N_train:-1])[0]) + '/' + str(np.shape(img_list)[0]) + '):')
        for filename in tqdm(img_list[N_train:-1]):
            # Read image
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            num_frame = os.path.split(filename)[1]

            # Create a mask with foreground pixels
            foreground_mask = (img-self.mean_px > alpha*(self.std_px + 2))
            foreground_mask = foreground_mask.astype(np.uint8)  # Convert to an unsigned byte
            foreground_mask*=255

            # Filter the noise
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, np.ones((5,5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

            # Get the number of connected components
            output = cv2.connectedComponentsWithStats(foreground_mask)
            (numLabels, _, stats, _) = output

            # Obtain bounding boxes
            frame_dets = []
            for i in range(1,numLabels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                frame_dets.append( BB(num_frame, i, 'car', x, y, x+w, y+h, 1) )
            detections.append(frame_dets)
            #cv2.imwrite('datasets/aicity/AICity_data/train/S03/c010/masks/mask_' + str(num_frame) + '.png', foreground_mask)
        return detections


    def test_adaptive(self, indices=None, color=False, alpha=1, rho=0.1):
        """
        Test the computed model using the adaptive method
        Params:
            indices: indicates the indices of the images to test. If unspecified
                     this will test for all the test images
            color: True: color images, False: grayscale
        Returns:
            List of lists of BBs [[BB,BB...],...]
        """
        print('Testing estimator:')

        # Get image list and number of images to use
        if indices is None:
            img_list = sorted(glob.glob(os.path.join(self.img_path,'frame_*.png')))
            N_train = math.floor(self.train_ratio*len(img_list))
            N_test = len(img_list) - N_train
        else: 
            img_list = sorted(glob.glob(os.path.join(self.img_path,'frame_*.png')))
            N_train = math.floor(self.train_ratio*len(img_list))
            N_test = len(img_list) - N_train

        # For all the images to test
        detections = []
        print('[1/1] Computing foreground masks for testing frames (' + str(np.shape(img_list[N_train:-1])[0]) + '/' + str(np.shape(img_list)[0]) + '):')
        for filename in tqdm(img_list[N_train:-1]):
            # Read image
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            num_frame = os.path.split(filename)[1]

            # Create a mask with foreground pixels
            foreground_mask = (img-self.mean_px > alpha*(self.std_px + 2))
            foreground_mask = foreground_mask.astype(np.uint8)  # Convert to an unsigned byte
            foreground_mask*=255

            # Filter the noise
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, np.ones((5,5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

            # Get the number of connected components
            output = cv2.connectedComponentsWithStats(foreground_mask)
            (numLabels, _, stats, _) = output

            frame_dets = []
            for i in range(1,numLabels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                frame_dets.append( BB(num_frame, i, 'car', x, y, x+w, y+h, 1) )
            detections.append(frame_dets)
            #cv2.imwrite('datasets/aicity/AICity_data/train/S03/c010/masks/mask_' + str(num_frame) + '.png', foreground_mask)

            # Update model
            # Does not work
            pixel_updates = img*(foreground_mask==0)
            self.mean_px = rho * pixel_updates + (1-rho) * self.mean_px
            var_px = rho * (pixel_updates-self.mean_px*(foreground_mask==0))**2 + (1-rho) * self.std_px**2
            self.std_px = np.sqrt(var_px)

        return detections