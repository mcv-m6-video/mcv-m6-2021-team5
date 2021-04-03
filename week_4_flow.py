import pyflow
import glob
from tqdm import tqdm
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
from utils.flow import *


def test_kitti_pair():
    # Flow Options:
    vis = False
    rs = None
    alpha = 0.006
    ratio = 0.5
    minWidth = 20
    nOuterFPIterations = 1
    nInnerFPIterations = 1
    nSORIterations = 7
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    for seq in ['000045', '000157']:

        im1 = Image.open('../datasets/others/image_0/'+seq+'_10.png')
        im2 = Image.open('../datasets/others/image_0/'+seq+'_10.png')
        gt_noc = Image.open('../datasets/flow/gt/flow_noc/'+seq+'_10.png')
        # gt_occ = Image.open('../datasets/flow/gt/flow_occ/'+seq+'_10.png')

        if rs is not None:
            im1 = im1.resize(rs)
            im2 = im2.resize(rs)

        im1 = np.array(im1)
        im2 = np.array(im2)

        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        
        msen, pepn, of_error1 = compute_of_metrics(flow, gt_noc)
        print("Sequence "+seq+"  -- MSEN: " + str(msen) + " | PEPN: " + str(pepn))

        if vis:
            hsv = np.zeros(im1.shape, dtype=np.uint8)
            hsv[:, :, 0] = 255
            hsv[:, :, 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            plt.figure(1)
            plt.subplot(311)
            plt.imshow(rgb)
            plt.subplot(312)
            plt.imshow(im2W[:, :, ::-1] * 255)
            plt.subplot(313)
            plt.imshow(im1 - im2W[:, :, ::-1])
            plt.show()

            #cv2.imwrite('../FLOW/flow/frame_'+str(i).zfill(4)+'.png', rgb)
            #cv2.imwrite('../FLOW/mc/frame_'+str(i).zfill(4)+'.png', im2W[:, :, ::-1] * 255)
            #cv2.imwrite('../FLOW/err/frame_'+str(i).zfill(4)+'.png', im1 - im2W[:, :, ::-1] * 255)


def test_kitti_full_seq():
    # Flow Options:
    vis = False
    rs = (640,360)
    alpha = 0.006
    ratio = 0.5
    minWidth = 20
    nOuterFPIterations = 1
    nInnerFPIterations = 1
    nSORIterations = 7
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    # Read the frames
    img_path = '../datasets/aicity/AICity_data/train/S03/c010/frames/'
    img_list = sorted(glob.glob(os.path.join(img_path,'frame_*.png')))

    # Compute optical flow for each frame pair
    for i in tqdm(range(len(img_list)-1)) :

        im1 = Image.open(img_list[i])
        im2 = Image.open(img_list[i+1])

        if rs is not None:
            im1 = im1.resize(rs)
            im2 = im2.resize(rs)

        im1 = np.array(im1)
        im2 = np.array(im2)

        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)

        if vis:
            hsv = np.zeros(im1.shape, dtype=np.uint8)
            hsv[:, :, 0] = 255
            hsv[:, :, 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite('../FLOW/flow/frame_'+str(i).zfill(4)+'.png', rgb)
            cv2.imwrite('../FLOW/mc/frame_'+str(i).zfill(4)+'.png', im2W[:, :, ::-1] * 255)
            cv2.imwrite('../FLOW/err/frame_'+str(i).zfill(4)+'.png', im1 - im2W[:, :, ::-1] * 255)

def aicity_gif_generation():
    # Flow Options:
    vis = False
    rs = (640,360)
    alpha = 0.006
    ratio = 0.5
    minWidth = 20
    nOuterFPIterations = 1
    nInnerFPIterations = 1
    nSORIterations = 7
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    # Read the frames
    img_path = '../datasets/aicity/AICity_data/train/S03/c010/frames/'
    img_list = sorted(glob.glob(os.path.join(img_path,'frame_*.png')))

    # Compute optical flow for each frame pair
    for i in tqdm(range(len(img_list)-1)) :

        im1 = Image.open(img_list[i])
        im2 = Image.open(img_list[i+1])

        if rs is not None:
            im1 = im1.resize(rs)
            im2 = im2.resize(rs)

        im1 = np.array(im1)
        im2 = np.array(im2)

        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)

        if vis:
            hsv = np.zeros(im1.shape, dtype=np.uint8)
            hsv[:, :, 0] = 255
            hsv[:, :, 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            plt.figure(1)
            plt.subplot(311)
            plt.imshow(rgb)
            plt.subplot(312)
            plt.imshow(im2W[:, :, ::-1] * 255)
            plt.subplot(313)
            plt.imshow(im1 - im2W[:, :, ::-1])
            plt.show()

            # cv2.imwrite('../FLOW/flow/frame_'+str(i).zfill(4)+'.png', rgb)
            # cv2.imwrite('../FLOW/mc/frame_'+str(i).zfill(4)+'.png', im2W[:, :, ::-1] * 255)
            # cv2.imwrite('../FLOW/err/frame_'+str(i).zfill(4)+'.png', im1 - im2W[:, :, ::-1] * 255)



def main():
    test_kitti_pair()

main()