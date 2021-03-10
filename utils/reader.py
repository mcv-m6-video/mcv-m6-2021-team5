import os
from collections import defaultdict, OrderedDict
from copy import deepcopy

import numpy as np
import xmltodict

from utils.bb import BB
#from src.tracking.track import Track


def parse_annotations(path):
    root, ext = os.path.splitext(path)

    if ext == ".xml":
        with open(path) as f:
            tracks = xmltodict.parse(f.read())['annotations']['track']

        annotations = []
        for track in tracks:
            id = track['@id']
            label = track['@label']
            boxes = track['box']
            for box in boxes:
                if label == 'car':
                    parked = box['attribute']['#text'].lower() == 'true'
                else:
                    parked = None
                annotations.append(BB(
                    frame=int(box['@frame']),
                    id=int(id),
                    label=label,
                    xtl=float(box['@xtl']),
                    ytl=float(box['@ytl']),
                    xbr=float(box['@xbr']),
                    ybr=float(box['@ybr']),
                    score=None
                ))
    
    if ext == ".txt":
        """
        MOTChallenge format [frame, ID, left, top, width, height, conf, -1, -1, -1]
        """

        with open(path) as f:
            lines = f.readlines()

        annotations = []
        for line in lines:
            data = line.split(',')
            annotations.append(BB(
                frame=int(data[0]) - 1,
                id=int(data[1]),
                label='car',
                xtl=float(data[2]),
                ytl=float(data[3]),
                xbr=float(data[2]) + float(data[4]),
                ybr=float(data[3]) + float(data[5]),
                score=float(data[6])
            ))


    return annotations


def group_by_frame(bboxes):
    grouped = defaultdict(list)
    for bb in bboxes:
        grouped[bb.frame].append(bb)
    return OrderedDict(sorted(grouped.items()))


def group_by_id(bboxes):
    grouped = defaultdict(list)
    for bb in bboxes:
        grouped[bb.id].append(bb)
    return OrderedDict(sorted(grouped.items()))


def group_in_tracks(bboxes, camera):
    grouped = group_by_id(bboxes)
    tracks = {}
    for id in grouped.keys():
        tracks[id] = Track(id, grouped[id], camera)
    return tracks


class AnnotationReader:

    """
    Creates AnnotationReader object that reads the annotations
    """
    def __init__(self, path):
        # Read XML file
        self.annotations = parse_annotations(path)
        self.classes = np.unique([bb.label for bb in self.annotations])

    def get_bboxes_per_frame(self, classes=None, noise_params=None):
        """
        This function returns the bounding boxes sorted by frame, in the format {frame: [BB, BB, BB...]}
        """

        if classes is None:
            classes = self.classes

        bboxes = []
        for bb in self.annotations:
            if bb.label in classes:  # filter by class
                current_box = deepcopy(bb)
                if noise_params:  # add noise
                    if np.random.random() > noise_params['drop']:
                        box_noisy = current_box.bbox + np.random.normal(noise_params['mean'], noise_params['std'], 4)
                        current_box.xtl = box_noisy[0]
                        current_box.ytl = box_noisy[1]
                        current_box.xbr = box_noisy[2]
                        current_box.ybr = box_noisy[3]
                        bboxes.append(current_box)
                else:
                    bboxes.append(current_box)

        bboxes = group_by_frame(bboxes)

        return bboxes


if __name__ == '__main__':
    reader = AnnotationReader(path='../../data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    # gt_noisy = reader.get_annotations(classes=['car'], noise_params={'drop': 0.05, 'mean': 0, 'std': 10})
    reader = AnnotationReader(path='../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    det = reader.get_annotations(classes=['car'])

    import cv2
    cap = cv2.VideoCapture('../../data/AICity_data/train/S03/c010/vdo.avi')
    frame = np.random.randint(0, len(gt))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    for d in gt[frame]:
        cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 255, 0), 2)
    for d in det[frame]:
        cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
