import argparse
import os
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import time

import config_folder as cf
from data_loaders.Chairs import Chairs
from data_loaders.kitti import KITTI
from data_loaders.sintel import Sintel
from model import MaskFlownet, MaskFlownet_S, Upsample, EpeLossWithMask


def centralize(img1, img2):
    rgb_mean = torch.cat((img1, img2), 2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
    return img1 - rgb_mean, img2-rgb_mean, rgb_mean

def read_of(flow_path):
    flow_raw = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED).astype(np.double)
    flow_u = (flow_raw[:,:,2] - 2**15) / 64.0
    flow_v = (flow_raw[:,:,1] - 2**15) / 64.0
    flow_valid = flow_raw[:,:,0] == 1
    flow_u[~flow_valid] = 0
    flow_v[~flow_valid] = 0
    return np.stack((flow_u, flow_v, flow_valid), axis=2)

def compute_of_metrics(flow, gt):
    square_error_matrix = (flow[:,:,0:2] - gt[:,:,0:2]) ** 2
    square_error_matrix_valid = square_error_matrix*np.stack((gt[:,:,2],gt[:,:,2]),axis=2)
    non_occluded_pixels = np.sum(gt[:,:,2] != 0)
    pixel_error_matrix = np.sqrt(np.sum(square_error_matrix_valid, axis= 2))
    msen = (1/non_occluded_pixels) * np.sum(pixel_error_matrix)
    erroneous_pixels = np.sum(pixel_error_matrix > 3)
    pepn = erroneous_pixels/non_occluded_pixels
    return msen, pepn, pixel_error_matrix

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='model checkpoint to load')
parser.add_argument('-b', '--batch', type=int, default=1,
                    help='Batch Size')
parser.add_argument('-f', '--root_folder', type=str, default=None,
                    help='Root folder of KITTI')
parser.add_argument('--resize', type=str, default='')
args = parser.parse_args()
resize = (int(args.resize.split(',')[0]), int(args.resize.split(',')[1])) if args.resize else None
num_workers = 2

with open(os.path.join('config_folder', args.dataset_cfg)) as f:
    config = cf.Reader(yaml.load(f))
with open(os.path.join('config_folder', args.config)) as f:
    config_model = cf.Reader(yaml.load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = eval(config_model.value['network']['class'])(config)
checkpoint = torch.load(os.path.join('weights', args.checkpoint))

net.load_state_dict(checkpoint)
net = net.to(device)


# Read images
img0 = np.array(Image.open('/content/000045_10.png'))
img1 = np.array(Image.open('/content/000045_11.png'))
im0 = torch.cuda.FloatTensor(np.expand_dims(img0/255.,0)).to(device)
im1 = torch.cuda.FloatTensor(np.expand_dims(img1/255.,0)).to(device)
gt_noc = read_of('/content/000045_10_gt.png')

with torch.no_grad():
  im0 = im0.permute(0, 3, 1, 2)
  im1 = im1.permute(0, 3, 1, 2)
  im0, im1, _ = centralize(im0, im1)

  shape = im0.shape
  pad_h = (64 - shape[2] % 64) % 64
  pad_w = (64 - shape[3] % 64) % 64
  if pad_h != 0 or pad_w != 0:
    im0 = F.interpolate(im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
    im1 = F.interpolate(im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')

  im0 = im0.to(device)
  im1 = im1.to(device)
  tic = time.time()
  pred, flows, warpeds = net(im0, im1)

  up_flow = Upsample(pred[-1], 4)
  up_occ_mask = Upsample(flows[0], 4)

  if pad_h != 0 or pad_w != 0:
    up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
              torch.tensor([shape[d] / up_flow.shape[d] for d in (2, 3)], device=device).view(1, 2, 1, 1)
    up_occ_mask = F.interpolate(up_occ_mask, size=[shape[2], shape[3]], mode='bilinear')
  toc = time.time()

  final_flow = up_flow.flip(1).to('cpu')[0]
  final_flow = np.array(final_flow.permute(1,2,0))
  hsv = np.zeros(img0.shape, dtype=np.uint8)
  hsv[..., 1] = 255
  mag, ang = cv2.cartToPolar(final_flow[..., 0], final_flow[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  fig = plt.figure()
  plt.imshow(rgb)
  fig.savefig('/content/flow.jpg')
  plt.show()

  msen, pepn, _ = compute_of_metrics(final_flow, gt_noc)
  print("MaskFlowNet: -- Time: " + str(toc-tic) + " | MSEN: " + str(msen) + " | PEPN: " + str(pepn))