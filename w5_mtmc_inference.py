import torch, torchvision
import detectron2
import numpy as np
import os, cv2, random
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from tqdm import tqdm
import pickle as pkl

from detectron2_tools.io import detectronReader
from utils.plotting import plot_detections, plot_detections2
from evaluation.iou import iou_bbox, compute_bb_distance
from evaluation.ap import mean_average_precision
from utils.reader import AnnotationReader
from tracking.tracking import track_max_overlap, track_kalman 
from utils.bb import *
import random
import time
from sklearn import manifold
import scipy
import math
import motmetrics as mm
from sklearn.decomposition import PCA

# Good threshold values for different models
# th_sd_veri = 1.3
# th_sd_ours = 4.62

write_images = False             # Write inference images to disk
#distance = 'b'                  # Use Bhattacharaya distance
#distance = 'lp'                 # Use Gaussian probability of match
distance = 'sd'                  # Use Squared embedding distance
th_lp = 0.5                      # Specific threshold for each distance
th_sd = 4.62
th_b = 23
use_pca = False                  # Reduce embedding dimensionality

# Sequences to use
cams =['c010','c011','c012','c013','c014','c015']


# Load the single camera tracks
tracks_dict = {}
frame_limits = {}
for cam in cams:
    with open('datasets/tracks/ourmodel_th085_rgb/tracks_seq_'+cam+'_085.pkl', 'rb') as f:
        tracks_dict[cam] = pkl.load(f)
        frame_limits[cam] = (int(tracks_dict[cam][0][0].frame),int(tracks_dict[cam][-1][0].frame))

# Get the frame boundaries
init_frame = np.min( [frame_limits[cam][0] for cam in cams] )
last_frame = np.max( [frame_limits[cam][1] for cam in cams] )
num_frames = last_frame-init_frame

# Load GT for each camera
gt_dict = {}
for cam in cams:
    gt_reader = AnnotationReader('datasets/aicity/AICity_data/train/S03/'+cam+'/gt/gt.txt')
    #gt_reader = AnnotationReader('datasets/aic19-track1-mtmc-train/train/S03/'+cam+'/gt/gt.txt')
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])
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
    gt_dict[cam] = bb_gt

#Load all feature vectors to fit a PCA
if use_pca:
    embeddings_list = []
    for i in range(0, num_frames):
        frame_number = init_frame + i
        for cam in cams:
            if frame_number < frame_limits[cam][0] or frame_number > frame_limits[cam][1]:
                continue

            # Get the tracks at the corresponding frame
            cam_frame_idx = frame_number - frame_limits[cam][0]
            tracks = tracks_dict[cam][ cam_frame_idx ]

            for t in tracks:
                embeddings_list.append(t.feature_vec)
    embeddings_vec = np.array(embeddings_list)
    embeddings_vec = np.squeeze(embeddings_vec)
    pca = PCA(n_components=64)
    pca.fit(embeddings_vec)

# Create the tracklet database
tracklets = {}
global_id = 0
# At each frame, get all the cameras and add them to the database
for i in range(0, num_frames):
    frame_number = init_frame + i
    for cam in cams:
        # Check if frame number is between the limits of each camera
        if frame_number < frame_limits[cam][0] or frame_number > frame_limits[cam][1]:
            continue

        # Get the tracks at the corresponding frame
        cam_frame_idx = frame_number - frame_limits[cam][0]
        tracks = tracks_dict[cam][ cam_frame_idx ]

        # Update the corresponding tracklet and compute the model
        for t in tracks:
            if use_pca:
                t.feature_vec = pca.transform(t.feature_vec)
            if get_id(t) not in tracklets.keys(): 
                tl = Tracklet(get_id(t), t)
                tracklets[get_id(t)] = tl
                tl.global_id = global_id
                global_id += 1
            else: 
                tracklets[get_id(t)].update_gmm(t)


#Filter out parked cars from the tracklets
tracklet_list = list(tracklets.values())
num_tracklets = len(tracklet_list)

filtered_tracklets=[]
for tracklet in tracklet_list:
    if np.median(tracklet.movement_list) > 2.75:
        filtered_tracklets.append(tracklet)
tracklet_list = filtered_tracklets

tracklet_list.sort(key=get_f_ini)
num_tracklets = len(tracklet_list)

# Run matching algorithm for all the tracklets
untracked = []
tracked = []
id_cnt = 0
for tracklet in tracklet_list: 
    idx_untracked = -1
    idx_tracked = -1

    logprobs = []
    b_distances = []
    s_distances = []
    for un in untracked:
        #Compute smallest distance with untracked tracklets
        if distance == 'lp':
            if tracklet.camera == un.camera:
                logprobs.append(0.0)
            else:
                lp = logprob(tracklet.gmm_mu, tracklet.gmm_var, un.gmm_mu)
                logprobs.append(lp)
            
        elif distance == 'b':
            if tracklet.camera == un.camera:
                b_distances.append(10000)
            else:
                bd = bhattacharyya(tracklet.gmm_mu, tracklet.gmm_var, un.gmm_mu, un.gmm_var)
                b_distances.append(bd)
        
        elif distance == 'sd':
            if tracklet.camera == un.camera:
                s_distances.append(10000)
            else:
                sd = squared_distance(tracklet.gmm_mu, un.gmm_mu)
                s_distances.append(sd)

    #Keep smallest distance and index
    if distance == 'lp' and logprobs:
        idx = np.argmax(logprobs)
        if logprobs[idx] > th_lp:
            idx_untracked, min_untracked = idx, logprobs[idx]

    elif distance == 'b' and b_distances:
        idx = np.argmin(b_distances)
        if b_distances[idx] < th_b:
            idx_untracked, min_untracked = idx, b_distances[idx]

    elif distance == 'sd' and s_distances:
        idx = np.argmin(s_distances)
        if s_distances[idx] < th_sd:
            idx_untracked, min_untracked = idx, s_distances[idx]

    logprobs = []
    b_distances = []
    s_distances = []
    for tr in tracked:
        #Compute smallest distance with already tracked tracklets
        if distance == 'lp':
            if tracklet.camera == un.camera:
                logprobs.append(0.0)
            else:
                lp = logprob(tracklet.gmm_mu, tracklet.gmm_var, tr.gmm_mu)
                logprobs.append(lp)
        elif distance == 'b':
            if tracklet.camera == un.camera:
                b_distances.append(10000)
            else:
                bd = bhattacharyya(tracklet.gmm_mu, tracklet.gmm_var, tr.gmm_mu, tr.gmm_var)
                b_distances.append(bd)
        elif distance == 'sd':
            if tracklet.camera == un.camera:
                s_distances.append(10000)
            else:
                sd = squared_distance(tracklet.gmm_mu, un.gmm_mu)
                s_distances.append(sd)

    #Keep smallest distance
    if distance == 'lp' and logprobs:
        idx = np.argmax(logprobs)
        if logprobs[idx] > th_lp:
            idx_tracked, min_tracked = idx, logprobs[idx]

    elif distance == 'b' and b_distances:
        idx = np.argmin(b_distances)
        if b_distances[idx] < th_b:
            idx_tracked, min_tracked = idx, b_distances[idx]
    
    elif distance == 'sd' and s_distances:
        idx = np.argmin(s_distances)
        if s_distances[idx] < th_sd:
            idx_tracked, min_tracked = idx, s_distances[idx]

    #No match with either untracked or tracked
    if idx_untracked == -1 and idx_tracked == -1:
        untracked.append(tracklet)
    
    #Match with a tracked tracklet
    if idx_untracked == -1 and idx_tracked != -1:
        tracklet.global_id = tracked[idx_tracked].global_id #Assign id of most similar tracklet
        tracked.append(tracklet)
    
    #Match with an untracked tracklet
    if idx_untracked != -1 and idx_tracked == -1:

        matched_tracklet = untracked.pop(idx_untracked) #Subtracted matched tracklet from untracked list   
        
        # Assign a new global id according to count
        tracklet.global_id = id_cnt
        matched_tracklet.global_id = id_cnt
        id_cnt += 1

        #Add both tracklets to tracked list
        tracked.append(tracklet) 
        tracked.append(matched_tracklet)

    #Match with both an untracked and a tracked tracklet. Keep the best case
    if idx_untracked != -1 and idx_tracked != -1:
        if distance == 'lp' and min_tracked>min_untracked or distance == 'b' and min_tracked<min_untracked or distance == 'sd' and min_tracked<min_untracked:
            tracklet.global_id = tracked[idx_tracked].global_id
            tracked.append(tracklet)
            
        else:
            matched_tracklet = untracked.pop(idx_untracked)
            tracklet.global_id = id_cnt
            matched_tracklet.global_id = id_cnt
            id_cnt += 1
            tracked.append(tracklet)
            tracked.append(matched_tracklet)


# Update tracklet dict
tracklets = {}
for tl in tracked:
    tracklets[tl.local_id] = tl

## Evaluation
acc = mm.MOTAccumulator(auto_id=True)

#Video capture for each camera
if write_images:
    captures = {}
    for cam in cams:
        video_cap = cv2.VideoCapture('datasets/aicity/AICity_data/train/S03/'+cam+'/vdo.avi')
        #video_cap = cv2.VideoCapture('datasets/aic19-track1-mtmc-train/train/S03/'+cam+'/vdo.avi')
        video_cap.set(1,frame_limits[cam][0])
        captures[cam] = video_cap

#cams = ['c010','c014']
for i in tqdm(range(0, num_frames)):
    frame_number = init_frame + i
    image_dict = {}
    for cam in cams:
        image_dict[cam] = np.zeros((480,640,3))
        # Check if frame number is between the limits of each camera
        if frame_number < frame_limits[cam][0] or frame_number > frame_limits[cam][1]:
            continue

        # Get the tracks at the corresponding frame
        cam_frame_idx = frame_number - frame_limits[cam][0]
        tracks = tracks_dict[cam][ cam_frame_idx ]
        
        # Compare to gt
        frame_gt = gt_dict[cam][cam_frame_idx]

        # Save global tracks in list
        global_tracks = []
        for t in tracks:
            # Check the track and assign its corresponding global id
            if get_id(t) in tracklets.keys():   
                t_id = tracklets[get_id(t)].global_id
                if t_id != -1:
                    t.id = t_id
                    global_tracks.append(t)
        if write_images:
            success, img = captures[cam].read()
            if success:
                img = plot_detections2(img, global_tracks, None, show=False)
                cv2.imwrite('figures/mtmc/'+cam+'/frame_'+str(i).zfill(4)+'.jpg', img)
                image_dict[cam] = img
            else:
                image_dict[cam] = np.zeros((480,640,3))

        #Evaluation: Compute distaces and create id arrays
        gt_ids = [gt.id for gt in frame_gt]
        det_ids = [detection.id for detection in global_tracks]   

        distances = np.zeros((len(frame_gt), len(global_tracks)))
        for ii, gt in enumerate(frame_gt):
            for jj, detection in enumerate(global_tracks):
                distances[ii,jj] = compute_bb_distance(detection, gt)
        acc.update(gt_ids, det_ids, distances)
    
    if write_images:
        for cam in cams:
            image_dict[cam] = cv2.resize(image_dict[cam], (640,480))
        a = np.concatenate((image_dict['c010'], image_dict['c011'], image_dict['c012']), axis=1)
        b = np.concatenate((image_dict['c013'], image_dict['c014'], image_dict['c015']), axis=1)
        output_img = np.concatenate((a,b),axis=0)
        cv2.imwrite('/figures/mtmc/frame_'+str(i).zfill(4)+'.jpg', output_img)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr', 'precision', 'recall'], name='acc')
print(summary)