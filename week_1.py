from utils.reader import AnnotationReader
from evaluation.ap import mean_average_precision
from utils.plotting import plot_detections 
from evaluation.iou import compute_iou_over_time
from matplotlib import pyplot as plt

import cv2
import numpy as np
import math
from utils.flow import *

#Paths to images, OF estimations and GT
color1_path = "datasets/others/colored_0/000045_10.png"
flow1_path = "datasets/flow/results/LKflow_000045_10.png"
gt1_path = "datasets/flow/gt/flow_noc/000045_10.png"
color2_path = "datasets/others/colored_0/000157_10.png"
flow2_path = "datasets/flow/results/LKflow_000157_10.png"
gt2_path = "datasets/flow/gt/flow_noc/000157_10.png"

gt_path = 'datasets/aicity/ai_challenge_s03_c010-full_annotation.xml'
det_path = 'datasets/aicity/AICity_data/train/S03/c010/det/'

def task1_1(bb_gt):
    reader = AnnotationReader(gt_path)
    drop_values = [0, 0.3]

    maps = []
    for drop in drop_values:
        noise_params = {'drop': drop, 'mean': 0, 'std': 0}
        gt_with_noise = reader.get_bboxes_per_frame(classes=['car'], noise_params=noise_params)

        bb_noisy = []
        for frame in gt_with_noise.keys():
            bb_noisy.append(gt_with_noise[frame])

        map, _, _ = mean_average_precision(bb_gt, bb_noisy, confidence_score=False)
        maps.append(map)
    print(maps)

def task1_2(bb_gt, bb_rcnn, bb_ssd, bb_yolo):

    # MASK RCNN
    map, _, _ = mean_average_precision(bb_gt, bb_rcnn)
    print('Mask RCNN mAP: ' + str(map))

    # SSD 512
    map, _, _ = mean_average_precision(bb_gt, bb_ssd)
    print('SSD 512 mAP: ' + str(map))

    # YOLO 3
    map, _, _ = mean_average_precision(bb_gt, bb_yolo)
    print('YOLO 3 mAP: ' + str(map))

def task2(bb_gt, bb_rcnn, bb_ssd, bb_yolo):

    plot_detections(bb_gt[0], bb_yolo[0])

    iou_yolo, iou_vec_yolo = compute_iou_over_time(bb_gt, bb_yolo)
    iou_ssd, iou_vec_ssd = compute_iou_over_time(bb_gt, bb_ssd)
    iou_rcnn, iou_vec_rcnn = compute_iou_over_time(bb_gt, bb_rcnn)

    _, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.plot(iou_vec_yolo)
    ax1.set_xlabel("YOLO IOU evolution")
    ax1.set_ylim([0,1])

    ax2.plot(iou_vec_ssd)
    ax2.set_xlabel("SSD IOU evolution")
    ax2.set_ylim([0,1])

    ax3.plot(iou_vec_rcnn)
    ax3.set_xlabel("Rcnn IOU evolution")
    ax3.set_ylim([0,1])
    plt.show()

    print('Mean IOU YOLO: ' + str(iou_yolo))
    print('Mean IOU SSD: ' + str(iou_ssd))
    print('Mean IOU RCNN: ' + str(iou_rcnn))


def task3(flow1, gt1, color_img1, flow2, gt2, color_img2):
    """
    Given 2 sequences of OF prediciton + GT and the corresponding color images this function:
        - calculates and plots error metrics: MSEN, PEPN, SEN image, SEN histogram
        - represents both the estimation and the GT using the HSV space.
    """

    msen, pepn, of_error1 = compute_of_metrics(flow1, gt1)
    print("Sequence 45  -- MSEN: " + str(msen) + " | PEPN: " + str(pepn))
    msen, pepn, of_error2 = compute_of_metrics(flow2, gt2)
    print("Sequence 157 -- MSEN: " + str(msen) + "  | PEPN: " + str(pepn))

    dense_of_plot(flow1, color_img1, filename="flow1")
    dense_of_plot(gt1, color_img1, filename="gt1")
    dense_of_plot(flow2, color_img2, filename="flow2")
    dense_of_plot(gt2, color_img2, filename="gt2")

    plot_of_error(of_error1, filename="1")
    plot_of_error(of_error2, filename="2")

    hist_of_error(of_error1, mask=gt1[:,:,2], filename="1")
    hist_of_error(of_error2, mask=gt2[:,:,2], filename="2")

def task4(flow1, gt1, color_img1, flow2, gt2, color_img2):
    """
    Given 2 sequences of OF prediciton + GT and the corresponding color images this function:
        - generates an arrow plot of the OF on top of the color image
    """
    arrow_of_plot(flow1, color_img1, filename="flow1")
    arrow_of_plot(gt1, color_img1, filename="gt1")
    arrow_of_plot(flow2, color_img2, filename="flow2")
    arrow_of_plot(gt2, color_img2, filename="gt2")
    print('Find the output at the Figures directory')


def main():

    ## TASKS 1-2
    print('Reading the data...')
    reader = AnnotationReader(gt_path)
    gt = reader.get_bboxes_per_frame(classes=['car'])
    reader = AnnotationReader(det_path + 'det_mask_rcnn.txt')
    det_rcnn = reader.get_bboxes_per_frame(classes=['car'])
    reader = AnnotationReader(det_path + 'det_ssd512.txt')
    det_ssd = reader.get_bboxes_per_frame(classes=['car'])
    reader = AnnotationReader(det_path + 'det_yolo3.txt')
    det_yolo = reader.get_bboxes_per_frame(classes=['car'])

    bb_gt = []
    bb_rcnn = []
    bb_ssd = []
    bb_yolo = []
    for frame in gt.keys():
        bb_gt.append(gt[frame])
        bb_rcnn.append(det_rcnn[frame])
        bb_ssd.append(det_ssd[frame])
        bb_yolo.append(det_yolo[frame])

    print('\n\n------------ Task 1 ------------')
    task1_1(bb_gt)
 
    task1_2(bb_gt, bb_rcnn, bb_ssd, bb_yolo)

    print('\n\n------------ Task 2 ------------')
    task2(bb_gt, bb_rcnn, bb_ssd, bb_yolo)


    ## TASKS 3-4
    flow1 = read_of(flow1_path)
    gt1 = read_of(gt1_path)
    flow2 = read_of(flow2_path)
    gt2 = read_of(gt2_path)
 
    color_img1 = read_img(color1_path)
    color_img2 = read_img(color2_path)

    print('\n\n------------ Task 3 ------------')
    task3(flow1, gt1, color_img1, flow2, gt2, color_img2)

    print('\n\n------------ Task 4 ------------')
    task4(flow1, gt1, color_img1, flow2, gt2, color_img2)


main()