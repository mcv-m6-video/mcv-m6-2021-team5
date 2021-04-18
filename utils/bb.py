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


class Tracklet:

    def __init__(self, id, t):
        # Tracklet params
        self.local_id = id
        self.camera = t.camera
        self.global_id = -1
        self.N = 1.0
        self.frames = []

        # Normalize the features
        vec_norm = t.feature_vec[0]/np.max(t.feature_vec[0])

        # Tracklet model
        self.feature_vecs = []
        self.feature_vecs.append(vec_norm)
        self.frames.append(t.frame)
        self.gmm_std = np.zeros(np.shape(vec_norm))
        self.gmm_mu = vec_norm
        

    def update_gmm(self, t):
        # Update the frames
        self.frames.append(t.frame)
        self.N += 1

        # Update the GMM mean
        vec_norm = t.feature_vec[0]/np.max(t.feature_vec[0])
        self.gmm_mu = (self.gmm_mu*(self.N-1) + vec_norm)/self.N

        # Update the GMM variance (TODO: Online method)
        self.feature_vecs.append(vec_norm)
        self.gmm_std = np.var(self.feature_vecs, axis=0, ddof=1)
    
    def match_tracklet(self, global_id, tracklet_id):
        print("TO DO")