from utils.reader import AnnotationReader, ImageReader
from evaluation.ap import mean_average_precision
from utils.plotting import plot_detections 
from evaluation.iou import compute_iou_over_time
from matplotlib import pyplot as plt

import cv2
import numpy as np
import math 

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

    # Parameters
    start = 535
    end = 545

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

    
    # Parameter tuning
    alphas = [3, 5, 7, 9]
    rhos = [0.0005, 0.01, 0.025, 0.05, 0.1, 0.3, 0.5]

    mAps_adaptive = []
    mAps_gaussian = []
    for alpha in alphas:
        for rho in rhos:
            print("Experiment with alpha = " + str(alpha) + " and rho = " + str(rho))
            mask_path_new = 'datasets/aicity/AICity_data/train/S03/c010/masks_adaptive' + '_' + str(alpha) + '_' + str(rho) + '/'

            # Reinit gray model
            gestimator = GaussianBGEstimator(img_path, mask_path)
            gestimator.load_pretrained('models/gaussian.pkl')
            gestimator.create_mask_path(mask_path_new)
            bb_gea = gestimator.test_adaptive(alpha=alpha, rho=rho, vis=True, N_test_start = start, N_test_end = end)

            # Evaluate
            map, _, _ = mean_average_precision(bb_gt, bb_gea, method = 'area')
            print('Gaussian Adaptive estimator mAP: ' + str(map))
            mAps_adaptive.append(map)
        
        # Create gray model
        gestimator = GaussianBGEstimator(img_path, mask_path)
        gestimator.load_pretrained('models/gaussian.pkl')
        mask_path_new = 'datasets/aicity/AICity_data/train/S03/c010/masks_test' + '_' + str(alpha) + '/'
        gestimator.create_mask_path(mask_path_new)
        bb_ge = gestimator.test(alpha=alpha, vis=True, N_test_start = start, N_test_end = end)

        # Evaluate
        map, _, _ = mean_average_precision(bb_gt, bb_ge, method = 'area')
        print('Gaussian estimator mAP: ' + str(map))
        mAps_gaussian.append(map)

    print(mAps_adaptive)
    print(mAps_gaussian)


    """
    print('\n\n------------------- Task 1: Gaussian models -------------------')
    
    # Create gray model
    gestimator = GaussianBGEstimator(img_path, mask_path)
    gestimator.load_pretrained('models/gaussian.pkl')

    # Create color model
    color_gestimator = ColorGaussianBGEstimator(img_path, mask_path)
    color_gestimator.load_pretrained('models/rgb_independent.pkl')
    
    # Test gray adaptive and non-adaptive
    bb_ge = gestimator.test(vis=True, N_test_start = start, N_test_end = end)
    bb_gea = gestimator.test_adaptive(vis=True, N_test_start = start, N_test_end = end)
    
    # Test color adaptive and non-adaptive
    bb_ge_color = color_gestimator.test(vis=True, N_test_end = 2000)
    bb_gea_color = color_gestimator.test_adaptive(vis=True, N_test_end = 2000)


    # Evaluate results
    map, _, _ = mean_average_precision(bb_gt, bb_ge)
    print('Gaussian gray estimator mAP: ' + str(map))

    map, _, _ = mean_average_precision(bb_gt, bb_gea)
    print('Gaussian gray Adaptive estimator mAP: ' + str(map))

    map, _, _ = mean_average_precision(bb_gt, bb_ge_color)
    print('Gaussian color estimator mAP: ' + str(map))

    map, _, _ = mean_average_precision(bb_gt, bb_gea_color)
    print('Gaussian color Adaptive estimator mAP: ' + str(map))
    """

    """
    print('\n\n------------------- Task 2: State of the art evaluation -------------------')
    # State of the art evaluation
    ocv_estimators = OpenCVBGEstimators(img_path, train_ratio=0.25)
    ocv_estimators.train(models=['MOG2', 'KNN'])

    bb_ocv_mog = ocv_estimators.test(model='MOG2',N_test_start = start, N_test_end = end)
    map, _, _ = mean_average_precision(bb_gt, bb_ocv_mog)
    print('OCV bg subtraction MOG mAP: ' + str(map))  

    bb_ocv_knn = ocv_estimators.test(model='KNN',N_test_start = start, N_test_end = end)
    map, _, _ = mean_average_precision(bb_gt, bb_ocv_knn)
    print('OCV bg subtraction KNN mAP: ' + str(map))  
    """

    #print('Initialize GMM:')
    #gestimator = GaussianBGEstimator(img_path, mask_path, train_ratio=0.005, n_components=15   )
    #gestimator.init_GMM()
    #print(gestimator.GMM_weights)

    #print('Test GMM:')
    #bb_gmm = gestimator.test_GMM(vis=True, N_test_start = 535, N_test_end = 560)

    
main()

    