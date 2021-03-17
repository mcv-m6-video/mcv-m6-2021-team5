from utils.reader import AnnotationReader, ImageReader
from evaluation.ap import mean_average_precision
from utils.plotting import plot_detections 
from evaluation.iou import compute_iou_over_time
from matplotlib import pyplot as plt

import cv2
import numpy as np
import math 

f = open("results.txt", "w")

from bgestimation.gaussian_estimation import GaussianBGEstimator
from bgestimation.bgs_opencv import OpenCVBGEstimators
from bgestimation.color_gaussian_estimation import ColorGaussianBGEstimator

#Paths to images
gt_path = 'datasets/aicity/ai_challenge_s03_c010-full_annotation.xml'
img_path = 'datasets/aicity/AICity_data/train/S03/c010/frames/'
det_path = 'datasets/aicity/AICity_data/train/S03/c010/det/'
mask_path = 'datasets/aicity/AICity_data/train/S03/c010/masks_color/'

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

    print('\n\n------------------- Initialization -------------------')
    
    ## Parameters
    start = 535
    end = 2141

    # Read GT
    reader = AnnotationReader(gt_path)
    gt = reader.get_bboxes_per_frame(classes=['car','bike'])

    # Keep only 75% of the BB GT
    # Ignore some static cars
    bb_gt = []
    ignore_ids=[0,1,2,3,4,5,6,7,8]
    for frame in range(start, end):
        boxes = []
        for box in gt[frame]:
            if box.id not in ignore_ids:
                box.label = 'car'
                boxes.append(box)
        bb_gt.append(boxes)


    print('\n\n------------------- Tasks 1,2,4: Gaussian models -------------------')
    
    # Create gray model
    gestimator = GaussianBGEstimator(img_path, mask_path)
    gestimator.load_pretrained('models/gaussian.pkl')

    # Create color model
    color_gestimator = ColorGaussianBGEstimator(img_path, mask_path, color_space='crcb')
    color_gestimator.load_pretrained('models/crcb_independent.pkl')
    
    # Test gray adaptive and non-adaptive
    bb_ge = gestimator.test(vis=True, alpha=11, N_test_start = start, N_test_end = end)
    bb_gea = gestimator.test_adaptive(vis=False, alpha=5, rho=0.01, N_test_start = start, N_test_end = end)
    
    # Test color adaptive and non-adaptive
    bb_ge_color = color_gestimator.test(vis=True, alpha=2.5, N_test_end = end)
    bb_gea_color = color_gestimator.test_adaptive(vis=True, alpha=2.5, N_test_end = end)


    # Evaluate results
    
    map, _, _ = mean_average_precision(bb_gt, bb_ge, method='area')
    print('Gaussian gray estimator mAP: ' + str(map))
    
    map, _, _ = mean_average_precision(bb_gt, bb_gea, method="area")
    print('Gaussian gray Adaptive estimator mAP: ' + str(map))
    
    map, _, _ = mean_average_precision(bb_gt, bb_ge_color, method='area')
    print('Gaussian color estimator mAP: ' + str(map))

    map, _, _ = mean_average_precision(bb_gt, bb_gea_color, method='area')
    print('Gaussian color Adaptive estimator mAP: ' + str(map))
    

    print('\n\n------------------- Task 3: State of the art evaluation -------------------')
    
    
    # State of the art evaluation
    ocv_estimators = OpenCVBGEstimators(img_path, train_ratio=0.25)
    ocv_estimators.train(models=['MOG2', 'KNN', 'SG'])

    bb_ocv_sg = ocv_estimators.test(model='SG',N_test_start = start, N_test_end = end)
    map, _, _ = mean_average_precision(bb_gt, bb_ocv_sg, method="area")
    print('OCV bg subtraction SG mAP: ' + str(map))

    bb_ocv_mog = ocv_estimators.test(model='MOG2',N_test_start = start, N_test_end = end)
    map, _, _ = mean_average_precision(bb_gt, bb_ocv_mog)
    print('OCV bg subtraction MOG mAP: ' + str(map))  

    bb_ocv_knn = ocv_estimators.test(model='KNN',N_test_start = start, N_test_end = end)
    map, _, _ = mean_average_precision(bb_gt, bb_ocv_knn, method="area")
    print('OCV bg subtraction KNN mAP: ' + str(map)) 



main()