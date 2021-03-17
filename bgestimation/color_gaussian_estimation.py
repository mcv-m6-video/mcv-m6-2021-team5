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
from sklearn.mixture import GaussianMixture

class ColorGaussianBGEstimator:

    """
    Creates GaussianBGEstimator object that reads the images and creates the model
    Params:
        img_path: path to input frames directory
        mask_path: path to output masks directory
        train_ratio: percentage of frames to use as train set
    """
    def __init__(self, img_path, mask_path, train_ratio=0.25, independent = True, color_space='rgb'):
        self.img_path = img_path
        self.mask_path = mask_path

        # Create mask path
        if os.path.exists(mask_path):
            shutil.rmtree(mask_path)
        os.makedirs(mask_path)

        self.train_ratio = train_ratio

        # Set image list and number of images to use
        self.img_list = sorted(glob.glob(os.path.join(self.img_path,'frame_*.png')))
        self.N_train = math.floor(self.train_ratio*len(self.img_list))
        self.N_test_start = self.N_train
        self.N_test_end = len(self.img_list)

        #Set image size atributes
        img_size = np.shape(cv2.imread(self.img_list[0], cv2.IMREAD_COLOR))
        w,h, = img_size[0:2] 
        self.im_w = w
        self.im_h = h 

        #Set color parameters
        self.color_space = color_space
        self.independent = independent

        if self.color_space == 'rgb':
            self.n_channels = 3
        if self.color_space == 'crcb':
            self.n_channels = 2
            self.color_flag = cv2.COLOR_BGR2YCrCb
        if self.color_space == 'hs':
            self.n_channels = 2
            self.color_flag = cv2.COLOR_BGR2HSV
        if self.color_space == 'ab':
            self.n_channels = 2
            self.color_flag = cv2.COLOR_BGR2Lab
        if self.color_space == 'h':
            self.n_channels = 1
            self.color_flag = cv2.COLOR_BGR2HSV

    def load_pretrained(self, filename):
        with open(filename, 'rb') as f:
            self.mean_px, self.std_px = pkl.load(f)

    def save_trained(self, filename):
        with open(filename,'wb') as f:
            pkl.dump([self.mean_px, self.std_px], f)

    def train(self):
        """
        This function returns a numpy array of train images
        This assumes that all images have the same size 
        for speed purposes
        """

        print('Training estimator:')
        # Get image size with first image
        img_size = np.shape(cv2.imread(self.img_list[0], cv2.IMREAD_COLOR))
        w,h, = img_size[0:2]     

        # Two pass method: first compute mean then std
        print('[1/2] Computing mean for training frames [' + str(0) + '-' + str(self.N_test_start) + ']:')
        
        if self.independent:
            self.mean_px = np.zeros((w,h,self.n_channels))
            self.std_px = np.zeros((w,h,self.n_channels))

            for filename in tqdm(self.img_list[0:self.N_train]):
                img = cv2.imread(filename, cv2.IMREAD_COLOR)

                #Convert to the specified color space
                if self.color_space != 'rgb':
                    img = cv2.cvtColor(img, self.color_flag)
                    if self.color_space == 'hs':
                        img = img[:,:,:2]
                    elif self.color_space == 'h':
                        img = img[:,:,0]
                        img = np.reshape(img, (self.im_w, self.im_h, 1))
                    else:
                        img = img[:,:,1:] #For color spaces Lab and YCrCb where we only want to keep the chrominance

                self.mean_px += img
            self.mean_px /= self.N_train

        print('[2/2] Computing std for training frames (' + str(np.shape(self.img_list[0:self.N_train])[0]) + '/' + str(np.shape(self.img_list)[0]) + '):')
        if self.independent:
            for filename in tqdm(self.img_list[0:self.N_train]):
                img = cv2.imread(filename, cv2.IMREAD_COLOR)

               #Convert to the specified color space
                if self.color_space != 'rgb':
                    img = cv2.cvtColor(img, self.color_flag)
                    if self.color_space == 'hs':
                        img = img[:,:,:2]
                    elif self.color_space == 'h':
                        img = img[:,:,0]
                        img = np.reshape(img, (self.im_w, self.im_h, 1))
                    else:
                        img = img[:,:,1:] #For color spaces Lab and YCrCb where we only want to keep the chrominance

                self.std_px += (img - self.mean_px) * (img - self.mean_px)
            print(filename)
            self.std_px = np.sqrt(self.std_px/(self.N_train-1))

        return self.mean_px, self.std_px   

    def test(self, alpha=6, vis=False, N_test_start=None, N_test_end=None):
        """
        Test the computed model
        Params:
            N_test_start: if None, all images are tested
            N_test_end: if None, all images are tested
        Returns:
            List of lists of BBs [[BB,BB...],...]
        """
        print('Testing estimator:')

        # Costomize frames to test (if None, all images are tested)
        if N_test_start != None:
            self.N_test_start = N_test_start
        
        if N_test_end != None:
            self.N_test_end = N_test_end

        # For all the images to test
        detections = []
        print('[1/1] Computing foreground masks for testing frames [' + str(self.N_test_start) + '-' + str(self.N_test_end) + ']:')
        for filename in tqdm(self.img_list[self.N_test_start:self.N_test_end]):

            # Read image
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            frame_name = os.path.split(filename)[1].split(".")[0]
            frame_num = frame_name.split("_")[1]

            #Convert to the specified color space
            if self.color_space != 'rgb':
                img = cv2.cvtColor(img, self.color_flag)
                if self.color_space == 'hs':
                    img = img[:,:,:2]
                elif self.color_space == 'h':
                    img = img[:,:,0]
                    img = np.reshape(img, (self.im_w, self.im_h, 1))
                else:
                    img = img[:,:,1:] #For color spaces Lab and YCrCb where we only want to keep the chrominance

            if self.independent:
                # Create a mask with foreground pixels from each channel
                if self.color_space == 'hs':
                    foreground_masks=np.zeros((self.im_w,self.im_h,2), dtype=bool)
                    foreground_masks[:,:,1] = abs(img[:,:,1]-self.mean_px[:,:,1]) > alpha*(self.std_px[:,:,1] + 2)
                    hue_diff = abs(img[:,:,0]-self.mean_px[:,:,0])
                    corrected_hue_diff = np.minimum(179-hue_diff, hue_diff) #Since Hue is in a circular space, the difference needs to be corrected
                    foreground_masks[:,:,0] = corrected_hue_diff > alpha*(self.std_px[:,:,0] + 2) 

                elif self.color_space == 'h':
                    foreground_masks=np.zeros((self.im_w,self.im_h,1), dtype=bool)
                    hue_diff = abs(img[:,:,0]-self.mean_px[:,:,0])
                    corrected_hue_diff = np.minimum(179-hue_diff, hue_diff) #Since Hue is in a circular space, the difference needs to be corrected
                    foreground_masks[:,:,0] = corrected_hue_diff > alpha*(self.std_px[:,:,0] + 2) 

                else:
                    foreground_masks = (abs(img-self.mean_px) > alpha*(self.std_px + 2))

                # Merge the masks from the different channels
                foreground_mask = np.zeros((self.im_w,self.im_h), dtype=bool)
                for i in range(self.n_channels):
                    foreground_mask = np.logical_or(foreground_mask, foreground_masks[:,:,i])
                foreground_mask = foreground_mask.astype(np.uint8)  # Convert to an unsigned byte
                foreground_mask*=255

            # Denoise mask
            foreground_mask_denoised = denoise_mask(foreground_mask, method=3)

            # Save masks
            if vis:
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_raw.png', foreground_mask)
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_denoised.png', foreground_mask_denoised)
                
            # # Method 1: connected components
            # output = cv2.connectedComponentsWithStats(foreground_mask_denoised)
            # (numLabels, _, stats, _) = output

            # # Obtain bounding boxes
            # frame_dets = []
            # foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            # for i in range(1, numLabels):
            #     x = stats[i, cv2.CC_STAT_LEFT]
            #     y = stats[i, cv2.CC_STAT_TOP]
            #     w = stats[i, cv2.CC_STAT_WIDTH]
            #     h = stats[i, cv2.CC_STAT_HEIGHT]
            #     if w > 20 and h > 10:
            #         frame_dets.append( BB(int(frame_num), i, 'car', x, y, x+w, y+h, 1) )
            # detections.append(frame_dets)
            
            #Method 2: find contours
            contours, _ = cv2.findContours(foreground_mask_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            frame_dets = []
            foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            j = 1
            for con in contours:
                (x, y, w, h) = cv2.boundingRect(con)
                if w > 20 and h > 10:
                    frame_dets.append(BB(int(frame_num), None, 'car', x, y, x+w, y+h, 1))
                    j = j+1
            detections.append(frame_dets)

            # if vis:
            #     plot_detections(frame_dets)
        return detections

    def test_adaptive(self, alpha=3, rho=0.01, vis=False, N_test_start=None, N_test_end=None):
        """
        Test the computed model using the adaptive method
        Params:
            N_test_start: if None, all images are tested
            N_test_end: if None, all images are tested
        Returns:
            List of lists of BBs [[BB,BB...],...]
        """
        print('Testing estimator:')

        # Costomize frames to test (if None, all images are tested)
        if N_test_start != None:
            self.N_test_start = N_test_start
        
        if N_test_end != None:
            self.N_test_end = N_test_end

        # For all the images to test
        detections = []
        print('[1/1] Computing foreground masks for testing frames [' + str(self.N_test_start) + '-' + str(self.N_test_end) + ']:')
        for filename in tqdm(self.img_list[self.N_test_start:self.N_test_end]):
            
            # Read image
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            frame_name = os.path.split(filename)[1].split(".")[0]
            frame_num = frame_name.split("_")[1]

            #Convert to the specified color space
            if self.color_space != 'rgb':
                img = cv2.cvtColor(img, self.color_flag)
                if self.color_space == 'hs':
                    img = img[:,:,:2]
                elif self.color_space == 'h':
                    img = img[:,:,0]
                    img = np.reshape(img, (self.im_w, self.im_h, 1))
                else:
                    img = img[:,:,1:] #For color spaces Lab and YCrCb where we only want to keep the chrominance
            if self.color_space == 'hs':
                foreground_masks=np.zeros((self.im_w,self.im_h,2), dtype=bool)
                foreground_masks[:,:,1] = abs(img[:,:,1]-self.mean_px[:,:,1]) > alpha*(self.std_px[:,:,1] + 2)
                hue_diff = abs(img[:,:,0]-self.mean_px[:,:,0])
                corrected_hue_diff = np.minimum(179-hue_diff, hue_diff) #Since Hue is in a circular space, the difference needs to be corrected
                foreground_masks[:,:,0] = corrected_hue_diff > alpha*(self.std_px[:,:,0] + 2) 
            elif self.color_space == 'h':
                foreground_masks=np.zeros((self.im_w,self.im_h,1), dtype=bool)
                hue_diff = abs(img[:,:,0]-self.mean_px[:,:,0])
                corrected_hue_diff = np.minimum(179-hue_diff, hue_diff) #Since Hue is in a circular space, the difference needs to be corrected
                foreground_masks[:,:,0] = corrected_hue_diff > alpha*(self.std_px[:,:,0] + 2) 
            else:
                foreground_masks = (abs(img-self.mean_px) > alpha*(self.std_px + 2))
         
            # Merge the masks from the different channels
            foreground_mask = np.zeros((self.im_w,self.im_h), dtype=bool)
            for i in range(self.n_channels):
                foreground_mask = np.logical_or(foreground_mask, foreground_masks[:,:,i])
            foreground_mask = foreground_mask.astype(np.uint8)  # Convert to an unsigned byte
            foreground_mask*=255

            # Denoise mask
            foreground_mask_denoised = denoise_mask(foreground_mask, method=3)

            # # Method 1: connected components
            # output = cv2.connectedComponentsWithStats(foreground_mask_denoised)
            # (numLabels, _, stats, _) = output

            # # Obtain bounding boxes
            # frame_dets = []
            # foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            # for i in range(1, numLabels):
            #     x = stats[i, cv2.CC_STAT_LEFT]
            #     y = stats[i, cv2.CC_STAT_TOP]
            #     w = stats[i, cv2.CC_STAT_WIDTH]
            #     h = stats[i, cv2.CC_STAT_HEIGHT]
            #     if w > 20 and h > 10:
            #         frame_dets.append( BB(int(frame_num), i, 'car', x, y, x+w, y+h, 1) )
            #     cv2.rectangle(foreground_mask_bbs,(x,y),(x+w,y+h),(255,255,255),-1)
            # detections.append(frame_dets)
            
            #Method 2: find contours
            contours, _ = cv2.findContours(foreground_mask_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            frame_dets = []
            foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            j = 1
            for con in contours:
                (x, y, w, h) = cv2.boundingRect(con)
                if w > 20 and h > 10:
                    frame_dets.append(BB(int(frame_num), None, 'car', x, y, x+w, y+h, 1))
                    j = j+1
                cv2.rectangle(foreground_mask_bbs,(x, y),(x + w, y + h),(255,255,255),-1)
            detections.append(frame_dets)       
            # Save masks
            if vis:
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_raw_ad.png', foreground_mask)
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_denoised_ad.png', foreground_mask_denoised)
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_denoised_bbs_ad.png', foreground_mask_bbs)

            # Convert mask to boolean
            foreground_mask_bbs = foreground_mask_bbs > 0

            # print(foreground_mask_bbs)

            # if vis:
            #     plot_detections(frame_dets)

            # Update model
            fg_pixels = foreground_mask_denoised==True
            bg_pixels = foreground_mask_denoised==False
            fg_pixels = fg_pixels.astype(np.uint8)
            bg_pixels = bg_pixels.astype(np.uint8)

            bg_pixels_3d = np.zeros((self.im_w, self.im_h, self.n_channels))
            fg_pixels_3d = np.zeros((self.im_w, self.im_h, self.n_channels))
            for ch in range(self.n_channels):
                bg_pixels_3d[:,:,ch] = bg_pixels
                fg_pixels_3d[:,:,ch] = fg_pixels

            #fg_pixels = foreground_mask_denoised==1
            #bg_pixels = foreground_mask_denoised==0

            image_pixels_bg = img*(bg_pixels_3d)
            mean_pixels_bg = self.mean_px*(bg_pixels_3d)
            var_pixels_bg = self.std_px*self.std_px*(bg_pixels_3d)

            # Compute updated mean only for background pixels
            updated_mean = rho * image_pixels_bg + (1-rho) * mean_pixels_bg
            #self.mean_px = self.mean_px*(fg_pixels_3d) + updated_mean*(bg_pixels_3d)
            #self.mean_px[bg_pixels_3d] = updated_mean
            np.putmask(self.mean_px, bg_pixels_3d, updated_mean)
            
            # Compute updated std only for background pixels
            updated_dev = np.sqrt( rho * (image_pixels_bg-mean_pixels_bg)**2 + (1-rho) * var_pixels_bg)
            #self.std_px = self.std_px*(fg_pixels_3d) + updated_dev*(bg_pixels_3d)
            #self.std_px[bg_pixels_3d] = updated_dev
            np.putmask(self.std_px, bg_pixels_3d, updated_dev)

        return detections