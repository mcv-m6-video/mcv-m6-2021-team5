from detectron2.structures import BoxMode
from detectron2.structures import Instances, Boxes
import xml.etree.ElementTree as ET
import torch
import os
import math
from utils.bb import BB
import random
import numpy as np

class detectronReader():
    def __init__(self, xmlfile):
        # Parse XML file
        tree = ET.parse(xmlfile) 
        root = tree.getroot()
        image_path = os.path.split(xmlfile)[0] + '/AICity_data/train/S03/c010/frames/'

        # Read all the boxes from each track and sort by frame number
        detections = []
        for track in root[2:]:
            for box in track:
                det = [int(box.attrib['frame']),
                        track.attrib['label'],
                        int(track.attrib['id']),
                        float(box.attrib['xtl']),
                        float(box.attrib['ytl']),
                        float(box.attrib['xbr']),
                        float(box.attrib['ybr'])]
                detections.append(det)
        detections = sorted(detections, key=lambda x: x[0])

        # Create a dict for every frame
        self.dataset_dicts = []
        last_frame = -1
        for i, det in enumerate(detections):
            # If frame has changed, restart record
            if det[0] != last_frame or i==0:
                if i != 0:
                    self.dataset_dicts.append(record)
                last_frame = det[0]
                record = {}
                record["file_name"] = image_path + 'frame_' + str(det[0]+1).zfill(4) + '.png'
                record["image_id"] = det[0]
                record["annotations"] = []
                record["width"] = 1920
                record["height"] = 1080

            # Add box to annotations
            if det[1]!='bike':
                box = {}
                box["bbox"] = det[3:7]
                box["bbox_mode"] = BoxMode.XYXY_ABS
                box["category_id"] = 0 if det[1] == 'car' else 1 #3 if det[1] == 'car' else 2
                box["track"] = det[2]
                record["annotations"].append(box)

            if i==len(detections)-1:
                self.dataset_dicts.append(record)


    def get_dict_from_xml(self, mode, train_ratio=0.25, K=0):
        """
        Reads an input xml file and returns a 
        detectron2 formatted list of dictionaries
        """
        N = len(self.dataset_dicts)
        N_train = math.floor(N*train_ratio)

        # Set the range
        if K == 0:
            self.range_train = list(range(0,N_train))
            self.range_val = list(range(N_train,N))
            #self.range_val = list(range(N_train,N_train+10))
        elif K == 1:
            self.range_train = list(range(N_train,2*N_train))
            self.range_val = list(range(0,N_train)) + list(range(2*N_train,N))
            #self.range_val = list(range(0,10)) + list(range(2*N_train,2*N_train+10))
        elif K == 2:
            self.range_train = list(range(2*N_train,3*N_train))
            self.range_val = list(range(0,2*N_train)) + list(range(3*N_train,N))
        elif K == 3:
            self.range_train = list(range(3*N_train,N))
            self.range_val = list(range(0,3*N_train))
        elif K == 4:
            # Random data: 25% (train_ratio) for training
            lin = np.linspace(0,N-1,N)
            lin = list(lin.astype(int))
            random.shuffle(lin)
            self.range_train = list(lin[0:N_train])
            self.range_val = list(lin[N_train:N])
            #self.range_train = list(lin[0:10])
            #self.range_val = list(lin[N_train:N_train+10])
        else:
            print('Invalid K value, enter a K between 0 and 3')
            return

        if mode == 'train':
            return [self.dataset_dicts[i] for i in self.range_train]
        elif mode == 'val':
            return [self.dataset_dicts[i] for i in self.range_val]
        else:
            print('Invalid mode: either train or val')

    # def get_range_for_k(self, K, train_ratio=0.25):
    #     N = len(self.dataset_dicts)
    #     N_train = math.floor(N*train_ratio)

    #     # Set the range
    #     if K == 0:
    #         range_train = list(range(0,N_train))
    #         range_val = list(range(N_train,N)) # TODO: Just for testing (N)
    #     elif K == 1:
    #         range_train = list(range(N_train,2*N_train))
    #         range_val = list(range(0,N_train)) + list(range(2*N_train,N))
    #     elif K == 2:
    #         range_train = list(range(2*N_train,3*N_train))
    #         range_val = list(range(0,2*N_train)) + list(range(3*N_train,N))
    #     elif K == 3:
    #         range_train = list(range(3*N_train,N))
    #         range_val = list(range(0,3*N_train))
    #     else:
    #         print('Invalid K value, enter a K between 0 and 3')
    #         return
        
    #     return [self.dataset_dicts[i] for i in range_train], [self.dataset_dicts[i] for i in range_val]

    def detectron2converter(self, input_pred, coco=False):
        """
        Convert the detectron2 prediction format
        to ours to compute the mAP
        """
        
        output_pred = []
        frame_num = 0

        for pred in input_pred:
            #print("Inference for frame: " + str(int(self.range_val[frame_num])))
            print(pred["instances"])
            pred_classes = pred["instances"].pred_classes.to("cpu")
            pred_scores = pred["instances"].scores.to("cpu")
            pred_boxes = pred["instances"].pred_boxes.to("cpu")


            pred_boxes = list(pred_boxes)

            box_list = []
            for i in range(0, len(pred_classes)):
                if coco:
                    if pred_classes[i] == 2: # class 2 = car
                        box = BB(int(self.range_val[frame_num]), 0, 'car', float(pred_boxes[i][0]), float(pred_boxes[i][1]), float(pred_boxes[i][2]), float(pred_boxes[i][3]), pred_scores[i])       
                        box_list.append(box)
                else:
                    box = BB(int(self.range_val[frame_num]), 0, 'car', float(pred_boxes[i][0]), float(pred_boxes[i][1]), float(pred_boxes[i][2]), float(pred_boxes[i][3]), pred_scores[i])
                    box_list.append(box)

            output_pred.append(box_list)
            frame_num += 1

        return output_pred

def read_detections_file(filename):
    output = []
    with open(filename, 'r') as f:
        detections = f.readlines()

        # Create a dict for every frame
        dataset_dicts = []
        last_frame = -1
        for i, det in enumerate(detections):
            det = det.split(',')
            # If frame has changed, restart record
            if int(det[0]) != last_frame or i==0 or i==len(detections)-1:
                if i != 0:
                    record["instances"].set("pred_classes",torch.CharTensor(pred_classes))
                    record["instances"].set("scores",torch.Tensor(scores))
                    record["instances"].set("pred_boxes",Boxes(torch.Tensor(pred_boxes)))
                    dataset_dicts.append(record)
                last_frame = int(det[0])
                record = {}
                pred_boxes = []
                pred_classes = []
                scores = []
                record["instances"] = Instances((1920,1080))

            # Add box to instances
            pred_boxes.append([float(det[2]),float(det[3]),float(det[2])+float(det[4]),float(det[3])+float(det[5])])
            pred_classes.append(0)
            scores.append(float(det[6]))
        return dataset_dicts
