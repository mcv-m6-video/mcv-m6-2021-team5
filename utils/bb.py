import numpy as np
import math 

class BB:

    def __init__(self, frame, id, label, xtl, ytl, xbr, ybr, score):
        self.frame = frame
        self.id = id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.score = score
        self.missed = 0
        self.feature_vec = []
        self.camera = 0
        self.parked = False

    @property
    def bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def bbox_score(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr, self.score]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ybr - self.ytl)

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (int((self.xtl + self.xbr) / 2), int((self.ytl + self.ybr) / 2))

    def __str__(self):
        return f'frame={self.frame}, id={self.id}, label={self.label}, bbox={self.bbox}, confidence={self.score}'

    def update_bbox(self, new_bbox):
        self.xtl = new_bbox.xtl
        self.ytl = new_bbox.ytl
        self.xbr = new_bbox.xbr
        self.ybr = new_bbox.ybr
    
    def increase_missed_bbox(self):
        self.missed += 1

    def set_camera(self, cam):
        self.camera = cam


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

class Tracklet:

    def __init__(self, id, t):
        # Tracklet params
        self.local_id = id
        self.camera = t.camera
        self.global_id = -1
        self.N = 1.0
        self.frames = []
        self.last_bb = t.bbox
        self.movement_list = []
        self.parked = False

        # Normalize the features
        #vec_norm = t.feature_vec[0]/np.max(t.feature_vec[0])
        vec_norm = t.feature_vec[0]

        # Tracklet model
        self.feature_vecs = []
        self.feature_vecs.append(vec_norm)
        self.frames.append(t.frame)
        self.gmm_var = np.zeros(np.shape(vec_norm))
        self.gmm_mu = vec_norm
        self.gmm_M2 = np.zeros(np.shape(vec_norm))
        

    def update_gmm(self, t):
        # Update the frames
        self.frames.append(t.frame)
        self.N += 1

        #Normalize embedding vector
        vec_norm = t.feature_vec[0]/np.max(t.feature_vec[0])

        # Update the GMM mean
        # self.gmm_mu = (self.gmm_mu*(self.N-1) + vec_norm)/self.N

        # Update the GMM variance (TODO: Online method)
        # self.feature_vecs.append(vec_norm)
        # self.gmm_std = np.var(self.feature_vecs, axis=0, ddof=1)

        #Update mean, variance and M2 with Welford's algorithm
        delta = vec_norm - self.gmm_mu
        self.gmm_mu += delta / self.N
        delta2 = vec_norm - self.gmm_mu
        self.gmm_M2 += delta * delta2
        self.gmm_var = self.gmm_M2/self.N

        #Update movement list, check if the car is parked and update last_center
        # self.movement_list.append(np.linalg.norm(self.last_center-np.array(t.center), ord=2))
        # if self.N >= 10 and not self.parked:
        #     parked = True
        #     for d in self.movement_list[-10:]:
        #         if d>20:
        #             parked = False
        #     self.parked = parked
        # self.last_center = np.array(t.center)
        iou = iou_bbox(self.last_bb, t.bbox)
        print(iou)
        if iou > 0.7:
            self.parked = True
        self.last_bb = t.bbox


    
    def match_tracklet(self, global_id, tracklet_id):
        print("TO DO")


