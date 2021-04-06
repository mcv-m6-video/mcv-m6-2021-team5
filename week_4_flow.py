import pyflow
import glob
from tqdm import tqdm
import cv2
import numpy as np
import os
from PIL import Image
from utils.flow import *
import time
from week_4_bm import block_matching

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

xmlfile = "datasets/aicity/ai_challenge_s03_c010-full_annotation.xml" 

def calculate_video_of(dir, start=1, end=2141, direc='forward', blk=32, bor=16, met='template'):
    video_of = [] 
    if direc=='forward':
        video_of.append(None)
    for i in range(start,end):
        img1_path = dir + "frame_" + str(str(i).zfill(4)) + ".png"
        img2_path = dir + "frame_" + str(str(i+1).zfill(4)) + ".png"
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        of = block_matching(img1, img2, direc, blk, bor, met)
        video_of.append(of)
    if direc=='backward':
        video_of.append(None)
    return video_of

def task_1_2():
    for seq in ['000045']:
        # Read images
        img1 = cv2.imread('./datasets/flow/frames/'+seq+'_10.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('./datasets/flow/frames/'+seq+'_11.png', cv2.IMREAD_GRAYSCALE)
        gt_noc = read_of('./datasets/flow/gt/flow_noc/'+seq+'_10.png')
        im1 = np.atleast_3d(img1.astype(float) / 255.)
        im2 = np.atleast_3d(img2.astype(float) / 255.)

        # Pyflow
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 1 
        tic = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
        flow_pyflow = np.dstack((u, v))
        toc = time.time()
        t_pyflow = toc-tic

        # Pyflow (fast)
        alpha = 0.012
        ratio = 0.5
        minWidth = 20
        nOuterFPIterations = 1
        nInnerFPIterations = 1
        nSORIterations = 15
        colType = 1 
        tic = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
        flow_pyflow_fast = np.dstack((u, v))
        toc = time.time()
        t_pyflow_fast = toc-tic

        # Farneback (openCV)
        tic = time.time()
        flow_farneback = cv2.calcOpticalFlowFarneback(img1, img2, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
        toc = time.time()
        t_farneback = toc-tic
        
        # Compute metrics
        msen, pepn, _ = compute_of_metrics(flow_pyflow, gt_noc)
        print("Pyflow: -- Time: " + str(t_pyflow)  + " | MSEN: " + str(msen) + " | PEPN: " + str(pepn))

        msen, pepn, _ = compute_of_metrics(flow_pyflow_fast, gt_noc)
        print("Pyflow: -- Time: " + str(t_pyflow_fast)  + " | MSEN: " + str(msen) + " | PEPN: " + str(pepn))

        msen, pepn, _ = compute_of_metrics(flow_farneback, gt_noc)
        print("Farneback: -- Time: " + str(t_farneback) + " | MSEN: " + str(msen) + " | PEPN: " + str(pepn))
        
        pyflow_hsv = hsv_plot(flow_pyflow)
        pyflow_fast_hsv = hsv_plot(flow_pyflow_fast)
        farneback_hsv = hsv_plot(flow_farneback)
        maxx = (np.max(np.max(farneback_hsv)))
        minn = (np.min(np.min(farneback_hsv)))

        fig = plt.figure()
        plt.subplot(311)
        plt.imshow(np.array(pyflow_hsv))
        plt.subplot(312)
        plt.imshow(np.array(pyflow_fast_hsv))
        plt.subplot(313)
        plt.imshow(np.array(farneback_hsv))
        plt.show()

def task_3_1():
    print('TODO')
    #Load detections
    detections_filename = 'models/faster_rcnn_X_101_32x8d_FPN_3x_1_700_ours.pkl'
    with open(detections_filename, 'rb') as f:
            bb_det = pkl.load(f)

    # Load GT
    # Read GT in our format for evaluation
    gt_reader = AnnotationReader(xmlfile)
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])

    # Get GT for evaluation
    bb_gt = []
    start, end = 535, 2141
    for frame in range(start, end):
        boxes = []
        for box in gt[frame]:
            boxes.append(box)
        bb_gt.append(boxes)
    
    #Calculate OF
    img_dir = './datasets/aicity/AICity_data/train/S03/c010/frames/'
    direc = 'forward'
    blk = 32
    bor = 16
    met = "template"
    of = calculate_video_of(img_dir, start=start, end=end, direc=direc, blk=blk, bor=bor, met=met)

    #Track detected objects and compare to GT
    track_max_overlap_of(bb_det, bb_gt, of)

def main():
    # print('Starting task 1.2: SOTA Optical Flow...')
    # task_1_2()

    print('Starting task 3.1: Tracking with OF...')
    task_3_1()
main()