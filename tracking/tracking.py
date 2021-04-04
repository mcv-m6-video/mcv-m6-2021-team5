import glob
import os
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm
from utils.bb import BB
import pickle as pkl
from evaluation.iou import iou_bbox, compute_bb_distance
import motmetrics as mm
from tracking.sort import Sort
from utils.non_maximum_supression import apply_non_max_supression

def track_max_overlap(bb_det, bb_gt):
    targets = []
    track_id = 0
    acc = mm.MOTAccumulator(auto_id=True)

    for i, (frame_dets, gt_dets) in enumerate(tqdm(zip(bb_det, bb_gt))):

        img_path = './datasets/aicity/AICity_data/train/S03/c010/frames/frame_' + str(frame_dets[0].frame+1).zfill(4) + '.png'
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # dets = []
        # for det in frame_dets:
        #     if det.score > 0.7:
        #         dets.append(det)
        # frame_dets = dets

        # frame_dets = apply_non_max_supression(frame_dets, i, 0.9)

        new_targets = []
        if targets == []:
            # Store the targets of the first frame
            for detection in frame_dets:
                detection.id = track_id
                track_id += 1
            new_targets = frame_dets

        else:
            for detection in frame_dets:
                # For each new bbox, compute all the IOUs
                candidates = []

                for target in targets:
                    candidates.append(iou_bbox(detection.bbox, target.bbox))
        
                # If the maximum overlap is zero, it is a new detection
                if np.max(candidates)==0:
                    detection.id = track_id
                    track_id += 1
                    new_targets.append(detection)

                    # unique=True
                    # for nt in new_targets:
                    #     if t.track_id == nt.track_id:
                    #     unique=False
                    # if(unique):
                    #     new_targets.append(t)

                else:
                    # Already existing target, update box, keep id
                    best_match_index = np.argmax(candidates)
                    t = targets[best_match_index]
                    t.update_bbox(detection)
                    new_targets.append(t)

                # unique=True
                # for nt in new_targets:
                #     if t.track_id == nt.track_id:
                #     unique=False
                # if(unique):
                #     new_targets.append(t)

        #Draw the image and also put the id
        for t in new_targets:
            np.random.seed(t.id)
            c = list(np.random.choice(range(int(256)), size=3)) 
            color = (int(c[0]), int(c[1]), int(c[2]))
            cv2.rectangle(im, (int(t.xtl), int(t.ytl)), (int(t.xbr), int(t.ybr)), color=color, thickness=3) 
            cv2.putText(im,str(t.id), (int(t.xtl), int(t.ytl)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imwrite('figures/tracking/overlap/frame_' + format(i, '04d') + '.jpg', im) 
        
        #Update targets
        targets = new_targets

        #Compute distaces and create id arrays
        gt_ids = [gt.id for gt in gt_dets]
        det_ids = [detection.id for detection in new_targets]   

        distances = np.zeros((len(gt_dets), len(new_targets)))
        for i, gt in enumerate(gt_dets):
            for j, detection in enumerate(new_targets):
                distances[i,j] = compute_bb_distance(detection, gt)
        acc.update(gt_ids, det_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1'], name='acc')
    print(summary)
    return summary



def track_kalman(bb_det, bb_gt):
    mot_tracker = Sort(max_age=2300, min_hits=2, iou_threshold=0.35) #create instance of the SORT tracker
    acc = mm.MOTAccumulator(auto_id=True)

    for i, (frame_dets, gt_dets) in enumerate(tqdm(zip(bb_det, bb_gt))):

        img_path = './datasets/aicity/AICity_data/train/S03/c010/frames/frame_' + str(frame_dets[0].frame+1).zfill(4) + '.png'
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Update Kalman tracker and obtain new prediction
        # Dets should be an array of bboxes (x1,y1,x2,y2,score)
        #dets = np.array([det.bbox_score for det in frame_dets])
        dets = []
        for det in frame_dets:
            if det.score > 0.8:
                dets.append(det)
        dets = np.array([det.bbox_score for det in dets])

        trackers = mot_tracker.update(dets)

        # Draw the image and also put the id
        # for t in trackers:
        #     np.random.seed(int(t[4]))
        #     c = list(np.random.choice(range(int(256)), size=3)) 
        #     color = (int(c[0]), int(c[1]), int(c[2]))
        #     cv2.rectangle(im, (int(t[0]),int(t[1])), (int(t[2]), int(t[3])), color, 3) 
        #     cv2.putText(im,str(int(t[4])), (int(t[0]),int(t[1])), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # cv2.imwrite('figures/tracking/kalman/frame_' + format(i, '04d') + '.jpg', im) 

        #Compute distaces and create id arrays
        gt_ids = [gt.id for gt in gt_dets]
        det_ids = [ t[4] for t in trackers]   

        distances = np.zeros((len(gt_dets), len(trackers)))
        for i, gt in enumerate(gt_dets):
            for j, t in enumerate(trackers):
                detection = BB(0,t[4],'', t[0], t[1], t[2], t[3], 0)
                distances[i,j] = compute_bb_distance(detection, gt)
        acc.update(gt_ids, det_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1'], name='acc')
    print(summary)
    return summary
        
    