import torch, torchvision
import numpy as np
import os, cv2, random
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle as pkl

from utils.plotting import plot_detections
from evaluation.ap import mean_average_precision
from utils.reader import AnnotationReader
from tracking.tracking import track_max_overlap, track_kalman 
from utils.bb import BB
import random
import time

# seqs=['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
seqs=['c014']
#seqs=['c011']
detector='mask_rcnn' 
method='kalman'
dataset = "datasets/aic19-track1-mtmc-train"
#dataset = "datasets/aicity/AICity_data"
tracks_dict = {}

dir_tracks = "ourmodel_th095"

start = time.time()
print(start)
for seq in seqs: 
    #Load detections
    reader = AnnotationReader(dataset + '/train/S03/'+seq+'/det/det_'+detector+'.txt')
    det_rcnn = reader.get_bboxes_per_frame(classes=['car'])

    # Load GT
    gt_reader = AnnotationReader(dataset + '/train/S03/'+seq+'/gt/gt.txt')
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])

    # Get GT for evaluation
    start, end = list(gt.keys())[0], list(gt.keys())[-1]
    bb_gt = []
    for frame in range(start,end):
        boxes = []
        if frame not in gt.keys():
            bb_gt.append([])
            continue
        for box in gt[frame]:
            boxes.append(box)
        bb_gt.append(boxes)

    # Fix detections
    bb_det = []
    for frame in range(start,end):
        boxes = []
        if frame not in det_rcnn.keys():
            bb_det.append([])
            continue
        for box in det_rcnn[frame]:
            boxes.append(box)
        bb_det.append(boxes)

    #Track detected objects and compare to GT
    tic = time.time()
    tracks_dict[seq] = track_kalman(bb_det, bb_gt, max_age=2000, min_hits=4, iou_threshold=0.1, score_threshold=0.95, seq=seq, extract_descriptors=True, start_frame=start)
    toc = time.time()
    print('Seq: '+seq+'| Tracking took: ' + str(toc-tic))
    with open('datasets/tracks/' + dir_tracks +  '/tracks_seq_'+seq+'.pkl','wb') as f:
        pkl.dump(tracks_dict[seq], f)
    final = time.time()
    print("Time elapsed: " + str(final- start))

# with open('datasets/tracks/' + dir_tracks + '/tracks_dict.pkl','wb') as f:
#     pkl.dump(tracks_dict, f)