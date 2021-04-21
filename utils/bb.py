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


class Tracklet:

    def __init__(self, id, t):
        # Tracklet params
        self.local_id = id
        self.camera = t.camera
        self.global_id = -1
        self.N = 1.0
        self.frames = []
        self.last_center = np.array(t.center)
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
        self.movement_list.append(np.linalg.norm(self.last_center-np.array(t.center), ord=2))
        if self.N >= 10 and not self.parked:
            parked = True
            for d in self.movement_list[-10:]:
                if d>5:
                    parked = False
            self.parked = parked
        self.last_center = np.array(t.center)
        
    
    def match_tracklet(self, global_id, tracklet_id):
        print("TO DO")