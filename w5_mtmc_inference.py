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
import networkx as nx
from sklearn import manifold

# Function to assign a unique id to each node
def get_id(track):
    return "c" + str(track.camera) + "f" + str(t.frame) + "id" + str(track.id)

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

# Parameters
distance_thresh = 100

# Read the detections of all the cameras
with open('tracks_seq_c010.pkl', 'rb') as f:
    tracks_seq = pkl.load(f)

descriptors = []
targets = []
for frame in tracks_seq:
    for t in frame: 
        descriptors.append(t.feature_vec[0]/np.max(t.feature_vec[0]))
        targets.append(int(t.id))
classes = np.unique(targets)
print(classes)
tsne = manifold.TSNE(n_components=2, n_iter=3000).fit_transform(descriptors)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
for label in classes:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(targets) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    np.random.seed(int(label))
    c = list(np.random.choice(range(int(256)), size=3))
    color = np.array([int(c[2]), int(c[1]), int(c[0])]).reshape(1,3)
    
    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color/255.0, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()





# tracks_dict = {}
# tracks_dict['c010'] = tracks_seq

# cams =['c010']#,'c011','c012','c013','c014','c015']

# # Start with the camera with the lowest frame number
# start = np.min( [tracks_dict[cam][0][0].frame for cam in cams] )
# end = np.max( [tracks_dict[cam][-1][0].frame for cam in cams] )

# print(start)
# print(end)

# # Car database:
# db = nx.Graph()
# start=0
# end=20

# # At each frame, get all the cameras and add them to the database
# for i in range(start,end):
#     for cam in cams:
#         tracks = tracks_dict[cam][i]
#         for t in tracks: 
#             db.add_node(get_id(t), descriptor=t.feature_vec, frame=t.frame, bbox=t.bbox)
#             for node in db:
#                 d = compute_distance(t.feature_vec, db.nodes[node]["descriptor"])
#                 if d < 0.1:
#                     db.add_edge(get_id(t), node, distance=d)

# pos=nx.spring_layout(db)
# nx.draw_networkx(db,pos, with_labels=True)
# labels = nx.get_edge_attributes(db,'distance')
# nx.draw_networkx_edge_labels(db,pos,edge_labels=labels)
# plt.show()            