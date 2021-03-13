from utils.reader import AnnotationReader, ImageReader
from evaluation.ap import mean_average_precision
from utils.plotting import plot_detections 
from evaluation.iou import compute_iou_over_time
from matplotlib import pyplot as plt

import cv2
import numpy as np
import math 

from bgestimation.gaussian_estimation import GaussianBGEstimator

#Paths to images
gt_path = 'datasets/aicity/ai_challenge_s03_c010-full_annotation.xml'
img_path = 'datasets/aicity/AICity_data/train/S03/c010/frames/'
det_path = 'datasets/aicity/AICity_data/train/S03/c010/det/'
mask_path = 'datasets/aicity/AICity_data/train/S03/c010/masks/'

def task1_1(train, val, gt):
    """
    Task 1.1: Gaussian background estimation
    Params:
        train: list of training images
        val: list of validation images
        gt: list of ground truth annotations
    """
    print(train)
    print(np.shape(train))
    print(np.shape(val))



def main():

    ## TASKS 1-2
    print('\n\n------------------- Initialization -------------------')
    
    # Read GT
    reader = AnnotationReader(gt_path)
    gt = reader.get_bboxes_per_frame(classes=['car'])

    bb_gt = []
    for frame in gt.keys():
        bb_gt.append(gt[frame])

    # Create model and infer the results
    gestimator = GaussianBGEstimator(img_path, mask_path)
    gestimator.load_pretrained('models/gaussian.pkl')
    #bb_ge = gestimator.test()
    bb_gea = gestimator.test_adaptive()

    # Evaluate
    #map, _, _ = mean_average_precision(bb_gt, bb_ge)
    #print('Gaussian estimator mAP: ' + str(map))

    map, _, _ = mean_average_precision(bb_gt, bb_gea)
    print('Gaussian Adaptive estimator mAP: ' + str(map))

    



main()