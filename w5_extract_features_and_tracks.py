import torch, torchvision
import detectron2
import numpy as np
import os, cv2, random
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle as pkl

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2_tools.io import detectronReader
from utils.plotting import plot_detections
from evaluation.ap import mean_average_precision
from utils.reader import AnnotationReader
from tracking.tracking import track_max_overlap, track_kalman 
from utils.bb import BB
import random
import time

seqs=['c010','c011','c012','c013','c014','c015']
detector=['mask_rcnn'] 
method='kalman'
tracks_dict = {}

for seq in seqs: 
    #Load detections
    reader = AnnotationReader('datasets/aicity/AICity_data/train/S03/'+seq+'/det/det_'+detector+'.txt')
    det_rcnn = reader.get_bboxes_per_frame(classes=['car'])

    # Load GT
    gt_reader = AnnotationReader('datasets/aicity/AICity_data/train/S03/'+seq+'/gt/gt.txt')
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
    tracks_dict[seq] = track_kalman(bb_det, bb_gt, max_age=2000, min_hits=4, iou_threshold=0.1, score_threshold=0.95, seq=seq, extract_descriptors=True)
    toc = time.time()
    print('Seq: '+seq+'| Tracking took: ' + str(toc-tic))

with open('tracks_dict.pkl','wb') as f:
    pkl.dump(tracks_dict, f)