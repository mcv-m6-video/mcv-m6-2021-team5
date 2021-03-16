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

    start = 535
    end = 2141

    #bb_ge = gestimator.test(vis=True, N_test_start = start, N_test_end = end)
    #bb_gea = gestimator.test_adaptive(vis=True, N_test_start = start, N_test_end = end)

    # Read GT
    reader = AnnotationReader(gt_path)
    gt = reader.get_bboxes_per_frame(classes=['car','bike'])

    # Keep only 75% of the BB GT
    bb_gt = []
    #for frame in gt.keys():
    # Ignore some static cars
    ignore_ids=[0,1,2,3,4,5,6,7,8]
    for frame in range(start, end):
        boxes = []
        for box in gt[frame]:
            if box.id not in ignore_ids:
                box.label = 'car'
                boxes.append(box)
        bb_gt.append(boxes)

    """
    # Evaluate
    map, _, _ = mean_average_precision(bb_gt, bb_ge)
    print('Gaussian estimator mAP: ' + str(map))

    map, _, _ = mean_average_precision(bb_gt, bb_gea)
    print('Gaussian Adaptive estimator mAP: ' + str(map))
    """
    
    # State of the art evaluation
    ocv_estimators = OpenCVBGEstimators(img_path, train_ratio=0.25)
    ocv_estimators.train()

    bb_ocv_mog = ocv_estimators.test(model='MOG2',N_test_start = start, N_test_end = end)
    map, _, _ = mean_average_precision(bb_gt, bb_ocv_mog)
    print('OCV bg subtraction MOG mAP: ' + str(map))  

    """
    for i in range(0,len(bb_ocv_mog)):
        im = plot_detections(bb_ocv_mog[i], gt_list=bb_gt[i], show=False)
        cv2.imwrite('./IMAGES/EVAL/frame_'+str(535+i).zfill(4)+'.png', im)
    """

main()