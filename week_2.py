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
    
    # Create model and infer the results
    gestimator = GaussianBGEstimator(img_path, mask_path)
    gestimator.load_pretrained('models/gaussian.pkl')

    bb_ge = gestimator.test(vis=True, N_test_start = 535, N_test_end = 580)
    bb_gea = gestimator.test_adaptive(vis=True, N_test_start = 535, N_test_end = 580)

    # Read GT
    reader = AnnotationReader(gt_path)
    gt = reader.get_bboxes_per_frame(classes=['car'])

    # Keep only 75% of the BB GT
    bb_gt = []
    #for frame in gt.keys():
    for frame in range(gestimator.N_test_start, gestimator.N_test_end):
        bb_gt.append(gt[frame])

    # Evaluate
    map, _, _ = mean_average_precision(bb_gt, bb_ge)
    print('Gaussian estimator mAP: ' + str(map))

    map, _, _ = mean_average_precision(bb_gt, bb_gea)
    print('Gaussian Adaptive estimator mAP: ' + str(map))

    



main()