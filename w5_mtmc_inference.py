import torch, torchvision
import detectron2
import numpy as np
import os, cv2, random
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from tqdm import tqdm
import pickle as pkl

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2_tools.io import detectronReader
from utils.plotting import plot_detections, plot_detections2
from evaluation.iou import iou_bbox, compute_bb_distance
from evaluation.ap import mean_average_precision
from utils.reader import AnnotationReader
from tracking.tracking import track_max_overlap, track_kalman 
from utils.bb import BB, Tracklet
import random
import time
from sklearn import manifold
import scipy
import math
import motmetrics as mm
from sklearn.decomposition import PCA

# Function to assign a unique id to each tracklet
def get_id(track):
    return "c" + str(track.camera) + "id" + str(int(track.id))

def get_f_ini(track):
    return track.frames[0]

def compute_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

def squared_distance(gmm_mu1, gmm_mu2):
    return np.linalg.norm(gmm_mu1-gmm_mu2, ord=2)

def logprob(gmm_mu, gmm_var, x):
    # TODO: This is still not the logprob... testing things jeje
    return np.sum( np.exp(-((gmm_mu - x)**2)/(2*gmm_var)) )/len(gmm_mu)

def bhattacharyya(gmm_mu1, gmm_var1, gmm_mu2, gmm_var2):
    epsilon_1d = (gmm_var1+gmm_var2)/2
    epsilon = np.diag(epsilon_1d)
    if np.count_nonzero(epsilon_1d) != len(gmm_var1):
        return 1000
    inv_epsilon = np.diag(1/epsilon_1d)
    term1 = (1/8)*(gmm_mu1-gmm_mu2)@inv_epsilon@(gmm_mu1-gmm_mu2).T
    num = np.prod(epsilon_1d[epsilon_1d!=0])
    den = math.sqrt(np.prod(gmm_var1[gmm_var1!=0])*np.prod(gmm_var2[gmm_var2!=0]))
    term2 = (1/2)*math.log(num/den)
    term2 = max(term2, 0.0)
    #print('Term 2:' + str(term2))
    #print(term1+term2)
    return term1+term2

# # Parameters
# distance_thresh = 100

# # Read the detections of all the cameras
# with open('tracks_seq_c010.pkl', 'rb') as f:
#     tracks_seq = pkl.load(f)

# descriptors = []
# targets = []
# for frame in tracks_seq:
#     for t in frame: 
#         descriptors.append(t.feature_vec[0]/np.max(t.feature_vec[0]))
#         targets.append(int(t.id))
# classes = np.unique(targets)
# print(classes)
# tsne = manifold.TSNE(n_components=2, n_iter=3000).fit_transform(descriptors)

# # extract x and y coordinates representing the positions of the images on T-SNE plot
# tx = tsne[:, 0]
# ty = tsne[:, 1]

# tx = scale_to_01_range(tx)
# ty = scale_to_01_range(ty)

# # initialize a matplotlib plot
# fig = plt.figure()
# ax = fig.add_subplot(111)

# # for every class, we'll add a scatter plot separately
# for label in classes:
#     # find the samples of the current class in the data
#     indices = [i for i, l in enumerate(targets) if l == label]

#     # extract the coordinates of the points of this class only
#     current_tx = np.take(tx, indices)
#     current_ty = np.take(ty, indices)

#     np.random.seed(int(label))
#     c = list(np.random.choice(range(int(256)), size=3))
#     color = np.array([int(c[2]), int(c[1]), int(c[0])]).reshape(1,3)
    
#     # add a scatter plot with the corresponding color and label
#     ax.scatter(current_tx, current_ty, c=color/255.0, label=label)

# # build a legend using the labels we set previously
# ax.legend(loc='best')

# # finally, show the plot
# plt.show()

# Read the detections of all the cameras
#distance = 'b'
#distance = 'lp'
distance = 'sd'
use_pca = False
cams =['c010','c011','c012','c013','c014','c015']
th_lp = 0.5
th_sd = 3.1
th_b = 23
tracks_dict = {}
frame_limits = {}
for cam in cams:
    with open('datasets/tracks/ourmodel_th095/tracks_seq_'+cam+'.pkl', 'rb') as f:
        tracks_dict[cam] = pkl.load(f)
        frame_limits[cam] = (int(tracks_dict[cam][0][0].frame),int(tracks_dict[cam][-1][0].frame))

# Get the frame boundaries
init_frame = np.min( [frame_limits[cam][0] for cam in cams] )
last_frame = np.max( [frame_limits[cam][1] for cam in cams] )
num_frames = last_frame-init_frame

# Load GT for each camera
gt_dict = {}
for cam in cams:
    #gt_reader = AnnotationReader('datasets/aicity/AICity_data/train/S03/'+cam+'/gt/gt.txt')
    gt_reader = AnnotationReader('datasets/aic19-track1-mtmc-train/train/S03/'+cam+'/gt/gt.txt')
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
    pca = PCA(n_components=20)
    pca.fit(embeddings_vec)

# Create the database
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

        # Update the corresponding tracklet 
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


# Run matching algorithm 
distance_image = np.zeros((len(tracklets), len(tracklets)))
tracklet_list = list(tracklets.values())
tracklet_list.sort(key=get_f_ini)

untracked = []
tracked = []
id_cnt = 0
for tracklet in tracklet_list:
    #print(tracklet.frames[0])
    idx_untracked = -1
    idx_tracked = -1
    print('Untracked: ' + str(len(untracked)))
    print('Tracked: ' + str(len(tracked)))

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

#####
# for ii, query in enumerate(tracklet_list):
#     logprobs = []
#     b_distances = []
#     untracked = []
#     tracked = []
#     #print(query.frames[0])
#     for jj, tl in enumerate(tracklet_list[:ii]):
#         if query.camera == tl.camera:
#             b_distances.append(math.inf)
#             logprobs.append(0)
#             continue

#         if distance == 'lp':
#             lp = logprob(query.gmm_mu, query.gmm_var, tl.gmm_mu)
#             logprobs.append(lp)
#             distance_image[ii,jj] = lp
#         elif distance == 'b':
#             bd = bhattacharyya(query.gmm_mu, query.gmm_var, tl.gmm_mu, tl.gmm_var)
#             b_distances.append(bd)
#             distance_image[ii,jj] = min(bd, 200)

#     if distance == 'lp':
#         idx = np.argmax(logprobs)
#         if logprobs[idx] > 0.5:
#             #if query.camera == tracklet_list[idx].camera
#             query.global_id = tracklet_list[idx].global_id
#         else:
#             query.global_id = query.local_id

#     elif distance == 'b':
#         idx = np.argmin(b_distances)
#         if b_distances[idx] < 1000000000000:
#             query.global_id = tracklet_list[idx].global_id
#         #else:
#             #query.global_id = -1

# print(np.max(np.max(distance_image)))
# print(np.min(np.min(distance_image)))
# plt.hist(np.ravel(distance_image), bins='auto')
# plt.show()
# plt.imshow(distance_image, cmap='gray')
# plt.show()

## Evaluation
acc = mm.MOTAccumulator(auto_id=True)

#Video capture for each camera
captures = {}
for cam in cams:
    #video_cap = cv2.VideoCapture('datasets/aicity/AICity_data/train/S03/'+cam+'/vdo.avi')
    video_cap = cv2.VideoCapture('datasets/aic19-track1-mtmc-train/train/S03/'+cam+'/vdo.avi')
    video_cap.set(1,frame_limits[cam][0])
    captures[cam] = video_cap

cams = ['c010','c014']
for i in tqdm(range(0, num_frames)):
    frame_number = init_frame + i
    for cam in cams:
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
            t_id = tracklets[get_id(t)].global_id
            if t_id != -1:
                t.id = t_id
                global_tracks.append(t)

        success, img = captures[cam].read()
        if success and i%5==0:
            img = plot_detections2(img, global_tracks, frame_gt, show=False)
            cv2.imwrite('figures/mtmc/'+cam+'/frame_'+str(i).zfill(4)+'.jpg', img)
            # img_size = (756,512)
            # cv2.imshow(str(i), cv2.resize(img, img_size))
            # cv2.waitKey(0)

        #Evaluation: Compute distaces and create id arrays
        gt_ids = [gt.id for gt in frame_gt]
        det_ids = [detection.id for detection in global_tracks]   

        distances = np.zeros((len(frame_gt), len(global_tracks)))
        for ii, gt in enumerate(frame_gt):
            for jj, detection in enumerate(global_tracks):
                distances[ii,jj] = compute_bb_distance(detection, gt)
        acc.update(gt_ids, det_ids, distances)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr'], name='acc')
print(summary)