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

class GaussianBGEstimator:

    """
    Creates GaussianBGEstimator object that reads the images and creates the model
    Params:
        img_path: path to input frames directory
        mask_path: path to output masks directory
        train_ratio: percentage of frames to use as train set
    """
    def __init__(self, img_path, mask_path, train_ratio=0.25, n_components=10):
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

        # Set parameters for GMM
        self.n_components = n_components
        self.GMM_weights = np.zeros(n_components)
        self.GMM_means = np.zeros(n_components)
        self.GMM_variances = np.zeros(n_components)

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
        # Get image size with first image
        img_size = np.shape(cv2.imread(self.img_list[0], cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE))
        w,h = img_size[0:2]

        # Preallocate numpy array for speed
        if color:
            self.mean_px = np.zeros((w,h,3))
            self.std_px = np.zeros((w,h,3))
        else:
            self.mean_px = np.zeros((w,h))
            self.std_px = np.zeros((w,h))

        # Two pass method: first compute mean then std
        print('[1/2] Computing mean for training frames [' + str(self.N_test_start) + '-' + str(self.N_test_end) + ']:')
        for filename in tqdm(self.img_list[0:self.N_train]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            self.mean_px += img
        self.mean_px /= self.N_train

        print('[2/2] Computing std for training frames (' + str(np.shape(self.img_list[0:self.N_train])[0]) + '/' + str(np.shape(self.img_list)[0]) + '):')
        for filename in tqdm(self.img_list[0:self.N_train]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            self.std_px += (img - self.mean_px) * (img - self.mean_px)
        print(filename)
        self.std_px = np.sqrt(self.std_px/(self.N_train-1))

        return self.mean_px, self.std_px   

    def init_GMM(self, color=False):
        """
        This function returns a GMM estimation from the initial train frames
        This assumes that all images have the same size 
        for speed purposes
        """

        print('Initializing GMM:')
        # Get image size with first image
        img_size = np.shape(cv2.imread(self.img_list[0], cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE))
        w,h = img_size[0:2]

        #Allocate space for image intensities array
        init_data = np.zeros((self.N_train,w,h))

        #Prepare data to fit the GMM
        print('Initializing GMM from frames 0-' + str(self.N_train) + ':')
        i=0
        for filename in tqdm(self.img_list[0:self.N_train]):
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            init_data[i,:,:] = img
            i+=1
        init_data = init_data.reshape((self.N_train*w*h, 1)) #Flatten input data array
        print(np.shape(init_data))

        #Fit GMM 
        gm = GaussianMixture(n_components=self.n_components, covariance_type='spherical', max_iter=50).fit(init_data)
        self.GMM_weights = gm.weights_
        self.GMM_means = gm.means_
        self.GMM_variances = gm.covariances_

        return self.GMM_weights, self.GMM_means, self.GMM_variances
    
    def test(self, color=False, alpha=6, vis=False, N_test_start=None, N_test_end=None):
        """
        Test the computed model
        Params:
            color: True: color images, False: grayscale
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
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            frame_name = os.path.split(filename)[1].split(".")[0]
            frame_num = frame_name.split("_")[1]

            # Create a mask with foreground pixels
            foreground_mask = (abs(img-self.mean_px) > alpha*(self.std_px + 2))
            foreground_mask = foreground_mask.astype(np.uint8)  # Convert to an unsigned byte
            foreground_mask*=255

            # Denoise mask
            foreground_mask_denoised = denoise_mask(foreground_mask, method=3)

            # Save masks
            if vis:
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_raw.png', foreground_mask)
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_denoised.png', foreground_mask_denoised)
                
            # Method 1: connected components
            output = cv2.connectedComponentsWithStats(foreground_mask_denoised)
            (numLabels, _, stats, _) = output

            # Obtain bounding boxes
            frame_dets = []
            foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            for i in range(1,numLabels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                frame_dets.append( BB(int(frame_num), i, 'car', x, y, x+w, y+h, 1) )
                cv2.rectangle(foreground_mask_bbs,(x,y),(x+w,y+h),(255,255,255),-1)
            detections.append(frame_dets)

            foreground_mask_bbs = foreground_mask_bbs/255 

            # # Method 2: find contours
            # detections = []
            # contours, _ = cv2.findContours(foreground_mask_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # frame_dets = []
            # foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            # i = 0
            # for con in contours:
            #     (x, y, w, h) = cv2.boundingRect(con)
            #     if w > 20 and h > 10:
            #         frame_dets.append( BB(int(frame_num), i, 'car', x, y, x+w, y+h, 1) )
            #         i = i+1
            #     foreground_mask_bbs = cv2.rectangle(foreground_mask_bbs,(x,x+w),(y,y+h),(255,255,255),-1)
            # detections.append(frame_dets)

            # plt.imshow(foreground_mask_bbs)
            # plt.show()

            # if vis:
            #     plot_detections(frame_dets)
        return detections

    def test_adaptive(self, color=False, alpha=6, rho=0.01, vis=False, N_test_start=None, N_test_end=None):
        """
        Test the computed model using the adaptive method
        Params:
            color: True: color images, False: grayscale
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
            img = cv2.imread(filename, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            frame_name = os.path.split(filename)[1].split(".")[0]
            frame_num = frame_name.split("_")[1]

            # Create a mask with foreground pixels
            foreground_mask = (abs(img-self.mean_px) > alpha*(self.std_px + 2))
            foreground_mask = foreground_mask.astype(np.uint8)  # Convert to an unsigned byte
            foreground_mask*=255

            foreground_mask_denoised = denoise_mask(foreground_mask, method=3)

            # Save masks
            if vis:
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_raw_ad.png', foreground_mask)
                cv2.imwrite(self.mask_path + 'mask_' + str(frame_name) + '_denoised_ad.png', foreground_mask_denoised)
             
        
            # Method 1: connected components
            output = cv2.connectedComponentsWithStats(foreground_mask_denoised)
            (numLabels, _, stats, _) = output

            # Obtain bounding boxes
            frame_dets = []
            foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            for i in range(1,numLabels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                frame_dets.append( BB(int(frame_num), i, 'car', x, y, x+w, y+h, 1) )
                cv2.rectangle(foreground_mask_bbs,(x,y),(x+w,y+h),(255,255,255),-1)
            detections.append(frame_dets)

            foreground_mask_bbs = foreground_mask_bbs/255 

            # # Method 2: find contours
            # detections = []
            # contours, _ = cv2.findContours(foreground_mask_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # frame_dets = []
            # foreground_mask_bbs = np.zeros(np.shape(foreground_mask))
            # i = 0
            # for con in contours:
            #     (x, y, w, h) = cv2.boundingRect(con)
            #     if w > 20 and h > 10:
            #         frame_dets.append( BB(int(frame_num), i, 'car', x, y, x+w, y+h, 1) )
            #         i = i+1
            #     foreground_mask_bbs = cv2.rectangle(foreground_mask_bbs,(x,x+w),(y,y+h),(255,255,255),-1)
            # detections.append(frame_dets)

            # plt.imshow(foreground_mask_bbs)
            # plt.show()

            # if vis:
            #     plot_detections(frame_dets)

            # Update model
            fg_pixels = foreground_mask_bbs==1
            bg_pixels = foreground_mask_bbs==0
            fg_pixels = fg_pixels.astype(np.uint8)
            bg_pixels = bg_pixels.astype(np.uint8)

            #fg_pixels = foreground_mask_denoised==1
            #bg_pixels = foreground_mask_denoised==0

            image_pixels_bg = img*(bg_pixels)
            mean_pixels_bg = self.mean_px*(bg_pixels)
            var_pixels_bg = self.std_px*self.std_px*(bg_pixels)

            # Compute updated mean only for background pixels
            updated_mean = rho * image_pixels_bg + (1-rho) * mean_pixels_bg
            #self.mean_px = self.mean_px*(fg_pixels) + updated_mean*(bg_pixels)
            #self.mean_px[bg_pixels] = updated_mean
            np.putmask(self.mean_px, bg_pixels, updated_mean)
            
            # Compute updated std only for background pixels
            updated_dev = np.sqrt( rho * (image_pixels_bg-mean_pixels_bg)**2 + (1-rho) * var_pixels_bg)
            #self.std_px = self.std_px*(fg_pixels) + updated_dev*(bg_pixels)
            #self.std_px[bg_pixels] = updated_dev
            np.putmask(self.std_px, bg_pixels, updated_dev)

        return detections
    
    def test_GMM(self, color=False):
        return 0