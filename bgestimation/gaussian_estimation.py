import glob
import os
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm

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
            ?
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
        print('[1/1] Computing foreground masks for testing frames (' + str(np.shape(img_list[N_train:-1])[0]) + '/' + str(np.shape(img_list)[0]) + '):')
        for filename in tqdm(img_list[N_train:-1]):
            # Read image
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            num_frame = os.path.split(filename)[1]

            # Create a mask with foreground pixels
            foreground_mask = (img-self.mean_px > alpha*(self.std_px + 2))
            foreground_mask = foreground_mask.astype(np.uint8)  # Convert to an unsigned byte
            foreground_mask*=255

            cv2.imwrite('datasets/aicity/AICity_data/train/S03/c010/masks/mask_' + str(num_frame) + '.png', foreground_mask)


        # for i, filename in enumerate(img_list[N_train:-1]):
        #     print(i)
        #     print(filename)
        #     num_frame = os.path.split(filename)[1]
        #     print(os.path.split(filename)[1])
        #     img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        #     foreground = (img-self.mean_px > alpha*(self.std_px + 2))
        #     print(type(foreground))

        #     mask = foreground.astype(np.uint8)  #convert to an unsigned byte
        #     mask*=255
        #     """
        #     cv2.imshow('mask', mask)
        #     cv2.waitKey(0) 
        #     cv2.destroyAllWindows()
        #     """
        #     print(np.shape(mask))
        #     print(type(mask))
            
        #     #cv2.imwrite('mask.png', mask)
        #     """
        #     cv2.imshow("frame", foreground*255)
        #     cv2.waitKey(0) 
        #     cv2.destroyAllWindows()
        #     """
        #     masks_path = 'datasets/aicity/AICity_data/train/S03/c010/masks/'
        #     if os.path.exists(masks_path):
        #         shutil.rmtree(masks_path)
        #     os.makedirs(masks_path)

        #     cv2.imwrite('datasets/aicity/AICity_data/train/S03/c010/masks/mask_' + str(num_frame) + '.png', mask)
        #     break