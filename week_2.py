from utils.reader import AnnotationReader, ImageReader
from evaluation.ap import mean_average_precision
from utils.plotting import plot_detections 
from evaluation.iou import compute_iou_over_time
from matplotlib import pyplot as plt

import cv2
import numpy as np
import math 

#Paths to images
gt_path = 'datasets/aicity/ai_challenge_s03_c010-full_annotation.xml'
img_path = 'datasets/aicity/AICity_data/train/S03/c010/frames/'
det_path = 'datasets/aicity/AICity_data/train/S03/c010/det/'

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
    print('Reading the data...')
    reader = AnnotationReader(gt_path)
    gt = reader.get_bboxes_per_frame(classes=['car'])
    img_reader = ImageReader(img_path)
    train_imgs = img_reader.get_train(color=True)
    val_imgs = img_reader.get_val(color=True)

    bb_gt = []
    for frame in gt.keys():
        bb_gt.append(gt[frame])

    print('Done!')
    print('\n\n------------------- Task 1 -------------------')
    task1_1(train_imgs, val_imgs, bb_gt)




main()