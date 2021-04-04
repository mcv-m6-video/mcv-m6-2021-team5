import numpy as np
from utils.bb import BB

def apply_non_max_supression(detections, frame_number, overlapThresh):
    boxes = [bb.bbox_score for bb in detections]
    new_boxes = non_max_suppression_fast(np.array(boxes), overlapThresh)
    new_bb = []
    for box in new_boxes:
        bb = BB(frame_number, 0, 'car', float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]))       
        new_bb.append(bb)
    return new_bb

# Malisiewicz et al.
def non_max_suppression_fast(boxes: np.array, overlapThresh: float):

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    xtl = boxes[:,0]
    ytl = boxes[:,1]
    xbr = boxes[:,2]
    ybr = boxes[:,3]
    scores = boxes[:,4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (xbr - xtl + 1) * (ybr - ytl + 1)
    idxs = np.argsort(ybr)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(xtl[i], xtl[idxs[:last]])
        yy1 = np.maximum(ytl[i], ytl[idxs[:last]])
        xx2 = np.minimum(xbr[i], xbr[idxs[:last]])
        yy2 = np.minimum(ybr[i], ybr[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        candidate_idxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        score = max(scores[candidate_idxs])
        boxes[i,4] = score

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")