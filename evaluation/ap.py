import numpy as np
from .iou import iou_vectorized

def mean_average_precision(gt_list, pred_list, method="score", classes=["car"]):
    """
    Mean Average Precision calculation
    Parameters:
        gt_list: [[Detection,...],...]
        pred_list: [[Detection,...],...]
        confidence_score: indicates the method to compute the map
        classes: indicates the classes of the dataset
    """

    precs = []
    recs = []
    aps = []
    for c in classes:
        gt_list_class = [[det for det in boxlist if det.label == c] for boxlist in gt_list]
        pred_list_class = [[det for det in boxlist if det.label == c] for boxlist in pred_list]
        ap, prec, rec = average_precision(gt_list_class, pred_list_class, method=method)
        precs.append(prec)
        recs.append(rec)
        aps.append(ap)
    prec = np.mean(precs)
    rec = np.mean(recs)
    map = np.mean(aps)

    return map, prec, rec


def average_precision(gt_list, pred_list, method="score"):
    """
    Average Precision with or without confidence scores.
    Params:
        gt_list: [[Detection,...],...]
        pred_list: [[Detection,...],...]
    """

    pred_list = [(i, det) for i in range(len(pred_list)) for det in pred_list[i]]
    if len(pred_list) == 0:
        return 0

    if method == "score" :
        sorted_ind = np.argsort([-det[1].score for det in pred_list])
        pred_list_sorted = [pred_list[i] for i in sorted_ind]
        ap, prec, rec = voc_ap(gt_list, pred_list_sorted)
    elif method == "area":
        sorted_ind = np.argsort([-det[1].area for det in pred_list])
        pred_list_sorted = [pred_list[i] for i in sorted_ind]
        ap, prec, rec = voc_ap(gt_list, pred_list_sorted)
    else:
        n = 10
        precs = []
        recs = []
        aps = []
        for _ in range(n):
            shuffled_ind = np.random.permutation(len(pred_list))
            pred_list_shuffled = [pred_list[i] for i in shuffled_ind]
            ap, prec, rec = voc_ap(gt_list, pred_list_shuffled)
            precs.append(prec)
            recs.append(rec)
            aps.append(ap)
        prec = np.mean(precs)
        rec = np.mean(recs)
        ap = np.mean(aps)
    return ap, prec, rec


# Below code is adapted from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py

def voc_ap(gt_list, pred_list, ovthresh=0.5):
    """
    Average Precision as defined by PASCAL VOC (11-point tracking).
    Params:
        gt_list: [[Detection,...],...]
        pred_list: [Detection,...]
        ovthresh: overlap threshold.
    """

    class_recs = []
    npos = 0
    for R in gt_list:
        bbox = np.array([det.bbox for det in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs.append({"bbox": bbox, "det": det})

    image_ids = [det[0] for det in pred_list]
    BB = np.array([det[1].bbox for det in pred_list]).reshape(-1, 4)

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = iou_vectorized(BBGT, bb[None, :])
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # compute VOC AP using 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap, prec, rec