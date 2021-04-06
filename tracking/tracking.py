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
import flowpy
from evaluation.iou import iou_bbox, compute_bb_distance
import motmetrics as mm
from tracking.sort import Sort
from utils.non_maximum_supression import apply_non_max_supression
from utils.plotting import plot_detections

def remove_missed_targets(targets, max_misses=2):
    cleaned_targets = []
    for target in targets:
        if target.missed <= max_misses:
            cleaned_targets.append(target)
        else:
            print('Eliminating target!')
    return cleaned_targets

def predict_targets(targets, flow, momentum=3):
    predicted_targets = [] 
    for t in targets:
        box_flow = flow[int(t.ytl):int(t.ybr), int(t.xtl):int(t.xbr)]
        box_flow = momentum*box_flow
        avg_flow = np.mean(box_flow, axis=(0,1))
        predicted = BB(t.frame, t.id, t.label, t.xtl - avg_flow[1], t.ytl - avg_flow[0],
                       t.xbr - avg_flow[1], t.ybr - avg_flow[0], t.score)
        predicted_targets.append(predicted)
    return predicted_targets

def add_new_target(query, targets):
    for target in targets:
        if iou_bbox(query.bbox, target.bbox) > 0.95:
            return targets
    targets.append(query)
    return targets


def track_max_overlap(bb_det, bb_gt):
    targets = []
    track_id = 0
    acc = mm.MOTAccumulator(auto_id=True)

    for i, (frame_dets, gt_dets) in enumerate(tqdm(zip(bb_det, bb_gt))):

        img_path = './datasets/aicity/AICity_data/train/S03/c010/frames/frame_' + str(frame_dets[0].frame-1).zfill(4) + '.png'
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

def track_max_overlap_of(bb_det, bb_gt, iou_threshold=0.2):
    targets = []
    track_id = 0
    acc = mm.MOTAccumulator(auto_id=True)
    for i, (frame_dets, gt_dets) in enumerate(zip(bb_det, bb_gt)):

        print(str(gt_dets[0].frame+1).zfill(4))
        img_path = './datasets/aicity/AICity_data/train/S03/c010/frames/frame_' + str(gt_dets[0].frame+1).zfill(4) + '.png'
        flow = flowpy.flow_read('./datasets/aicity/AICity_data/train/S03/c010/of/frame_'+ str(gt_dets[0].frame+1).zfill(4) +'.png')
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)

        new_targets = []
        if targets == []:
            # Store the targets of the first frame
            for detection in frame_dets:
                detection.id = track_id
                track_id += 1
            new_targets = frame_dets

        else:
            #Remove targets that gave been missed multiple times
            targets = remove_missed_targets(targets, max_misses=4) 

            #Predict new targets using OF (always assuming forward direction)
            predicted_targets = predict_targets(targets, flow)
            
            for target in targets:  
                #Max overlap of detections and target over targets, update bbox
                candidates = []
                for detection in frame_dets:
                    candidates.append(iou_bbox(detection.bbox, target.bbox))
                
                # If the maximum overlap is not zero, already existing target, update box, keep id
                if np.max(candidates)>=iou_threshold:
                    best_match_index = np.argmax(candidates)
                    d = frame_dets[best_match_index]
                    target.update_bbox(d)
                    target.missed = 0
                    new_targets = add_new_target(target, new_targets)

                else:
                    #Max overlap of target and predictions
                    candidates = []
                    for prediction in predicted_targets:
                        candidates.append(iou_bbox(prediction.bbox, target.bbox))
                    
                    # If the maximum overlap is not zero, already existing target but missed in the detection
                    # Update bbox, keep id, increase misses counter
                    if np.max(candidates)>=iou_threshold:
                        best_match_index = np.argmax(candidates)
                        d = predicted_targets[best_match_index]
                        target.update_bbox(d)
                        target.increase_missed_bbox()
                        new_targets = add_new_target(target, new_targets)
            
            #Max overlap of detections and target over detections, if 0, new track
            for detection in frame_dets: 
                candidates = []
                for target in targets: 
                    candidates.append(iou_bbox(detection.bbox, target.bbox)) 
                pred_candidates = []
                for predicted in predicted_targets:
                    pred_candidates.append(iou_bbox(detection.bbox,predicted.bbox))

                if np.max(candidates)<iou_threshold and np.max(pred_candidates)<iou_threshold:
                    # New detection
                    detection.id = track_id
                    track_id += 1  
                    new_targets = add_new_target(detection, new_targets)             

        #Draw the image and also put the id
        for t in new_targets:
            np.random.seed(t.id)
            c = list(np.random.choice(range(int(256)), size=3)) 
            color = (int(c[0]), int(c[1]), int(c[2]))
            cv2.rectangle(im, (int(t.xtl), int(t.ytl)), (int(t.xbr), int(t.ybr)), color=color, thickness=3) 
            cv2.putText(im,str(t.id), (int(t.xtl), int(t.ytl)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imwrite('figures/tracking/overlap_of/frame_' + format(i, '04d') + '.jpg', im) 
        
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


        
    