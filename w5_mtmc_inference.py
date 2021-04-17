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


# Read the detections of all the sequences
with open('tracks_dict.pkl', 'rb') as f:
    tracks_dict = pkl.load(f)

