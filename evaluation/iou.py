import numpy as np

def iou_bbox(box1, box2):
    """
    Input format is [xtl, ytl, xbr, ybr] per bounding box, where
    tl and br indicate top-left and bottom-right corners of the bbox respectively
    """
    #determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union 
    iou = interArea / float(box1Area + box2Area - interArea)

    # return the intersection over union value
    return iou

def compute_iou_over_time(gt_list, pred_list):
    """
    Compute the mean IOU over time (for each frame)
    Parameters:
        gt_list: [[Detection,...],...]
        pred_list: [[Detection,...],...]
    """
    frame_ious = []
    for dets_gt, dets_pred in zip(gt_list, pred_list):
        box_ious = []
        for det in dets_gt:
            ious = []
            for det_p in dets_pred:
                iou = iou_bbox(det.bbox,det_p.bbox)
                ious.append(iou)
            box_ious.append(max(ious))
        frame_ious.append(np.mean(box_ious))
    mean_iou_global = np.mean(frame_ious)
    return mean_iou_global, frame_ious


def iou_vectorized(boxes1, boxes2):
    """
    Compute overlaps between two sets of boxes.
    Params:
        boxes1: [[xtl,ytl,xbr,ybr],...]
        boxes2: [[xtl,ytl,xbr,ybr],...]
    Returns:
        overlaps: matrix of pairwise overlaps.
    """

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # intersection
    ixmin = np.maximum(x11, np.transpose(x21))
    iymin = np.maximum(y11, np.transpose(y21))
    ixmax = np.minimum(x12, np.transpose(x22))
    iymax = np.minimum(y12, np.transpose(y22))
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    area1 = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    area2 = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)
    uni = area1 + np.transpose(area2) - inters

    overlaps = inters / uni

    return overlaps

def compute_bb_distance(box1, box2):
    center1 = box1.center
    center2 = box2.center
    return np.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)
