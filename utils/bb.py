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