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
from utils.plotting import plot_detections
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

def compute_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def logprob(gmm_mu, gmm_var, x):
    # TODO: This is still not the logprob... testing things jeje
    return np.sum( np.exp(-((gmm_mu - x)**2)/(2*gmm_var)) )/len(gmm_mu)

def bhattacharyya(gmm_mu1, gmm_var1, gmm_mu2, gmm_var2):
    t0 = time.time()
    epsilon_1d = (gmm_var1+gmm_var2)/2
    epsilon = np.diag(epsilon_1d)
    if np.count_nonzero(epsilon_1d) != len(gmm_var1):
        return 10000000
    inv_epsilon = np.diag(1/epsilon_1d)
    t1 = time.time()
    #print(t1-t0)
    term1 = (1/8)*(gmm_mu1-gmm_mu2)@inv_epsilon@(gmm_mu1-gmm_mu2).T
    #print('Term 1:' + str(term1))
    t2 = time.time()
    #print(t2-t1)
    t3 = time.time()
    num = math.sqrt(np.prod(gmm_var1[gmm_var1!=0])*np.prod(gmm_var2[gmm_var2!=0]))
    den = np.prod(epsilon_1d[epsilon_1d!=0])
    term2 = (1/2)*math.log(num/den)
    #print('Term 2:' + str(term2))
    #print(time.time()-t3)
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
cams =['c010','c011','c012','c013','c015']
tracks_dict = {}
frame_limits = {}
for cam in cams:
    with open('datasets/tracks/pretrained_VeRi_th095/tracks_seq_'+cam+'.pkl', 'rb') as f:
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
            if get_id(t) not in tracklets.keys():
                t.feature_vec = pca.transform(t.feature_vec)
                tl = Tracklet(get_id(t), t)
                tracklets[get_id(t)] = tl
                tl.global_id = global_id
                global_id += 1
            else:
                t.feature_vec = pca.transform(t.feature_vec)
                tracklets[get_id(t)].update_gmm(t)

# Run matching algorithm
prob_matrix = np.zeros((len(tracklets), len(tracklets)))
for ii, query in enumerate(tracklets.values()):
    print(ii)
    logprobs = []
    b_distances = []
    for jj, tl in enumerate(tracklets.values()):
        if query.camera == tl.camera:
            continue
        #lp = logprob(query.gmm_mu, query.gmm_var, tl.gmm_mu)
        bd = bhattacharyya(query.gmm_mu, query.gmm_var, tl.gmm_mu, tl.gmm_var)
        #logprobs.append(lp)
        b_distances.append(bd)
        #prob_matrix[ii,jj] = lp
        #prob_matrix[ii,jj] = bd
    #idx = np.argmax(logprobs)
    idx = np.argmin(b_distances)
    # if logprobs[idx] > 0.09:
    #     prob_matrix[ii,idx] = 1
    #     print('Match! tracklet ids:')
    #     print(tl.global_id)
    #     print(query.global_id)
    #     tl.global_id = query.global_id
    if b_distances[idx] < 23:
        #prob_matrix[ii,idx] = 1
        print('Match! tracklet ids:')
        print(tl.global_id)
        print(query.global_id)
        tl.global_id = query.global_id
    else:
        query.global_id = -1

#plt.imshow(prob_matrix, cmap='gray')
#plt.show()



## Evaluation
acc = mm.MOTAccumulator(auto_id=True)

# At each frame, get all the detections and assign the global id corresponding to their tracklet
for i in range(0, num_frames):
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

        #Evaluation: Compute distaces and create id arrays
        gt_ids = [gt.id for gt in frame_gt]
        det_ids = [detection.id for detection in global_tracks]   

        distances = np.zeros((len(frame_gt), len(global_tracks)))
        for i, gt in enumerate(frame_gt):
            for j, detection in enumerate(global_tracks):
                distances[i,j] = compute_bb_distance(detection, gt)
        acc.update(gt_ids, det_ids, distances)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1'], name='acc')
print(summary)