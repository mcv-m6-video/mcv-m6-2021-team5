import torch, torchvision
import detectron2
import numpy as np
import os, cv2, random
from matplotlib import pyplot as plt
import tqdm

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2_tools.io import detectronReader, detectron2converter
from evaluation.ap import mean_average_precision
from utils.reader import AnnotationReader


xmlfile = "datasets/aicity/ai_challenge_s03_c010-full_annotation.xml" 

def task_1_1():
    print('TODO')

def task_1_2():
    """
    This function reads the data and fine tunes an existing detectron2 model
    """
    # Read GT in our format for evaluation
    gt_reader = AnnotationReader(xmlfile)
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])

    # K=4 cross validation
    for k in range(0,4):

        # Init dataset reader
        reader = detectronReader(xmlfile)

        # Register the datasets for this iteration
        for d in ['train'+str(k), 'val'+str(k)]:
            DatasetCatalog.register("AICity_"+d, lambda d=d: reader.get_dict_from_xml(d[0:-1],K=k))
            MetadataCatalog.get("AICity_"+d).set(thing_classes=['car'])
        
        aicity_metadata = MetadataCatalog.get("AICity_train0")

        # Read the dataset from the xml file
        train_dict = reader.get_dict_from_xml('train',K=k)
        
        # Test if data is properly read
        for d in train_dict[0:1]:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=aicity_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            plt.imshow(out.get_image()[:, :, ::-1])
            plt.show()

        # Train
        cfg = get_cfg()
        cfg.OUTPUT_DIR = 'detectron2_models/faster_rcnn_X_101_32x8d_FPN_3x_KV_'+str(k)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("AICity_train"+str(k),)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  
        cfg.SOLVER.MAX_ITER = 300  
        cfg.SOLVER.STEPS = []        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 


        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Inference on the test dataset
        test_dict = reader.get_dict_from_xml('test',K=k)

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
 
        # Predict
        predictions = []
        for d in tqdm(test_dict):    
            im = cv2.imread(d["file_name"])
            outputs = predictor(im) 
            predictions.append(outputs)

        # Get GT for evaluation
        bb_gt = []
        for frame in range(start, end):
            boxes = []
            for box in gt[frame]:
                boxes.append(box)
            bb_gt.append(boxes)


        # Compute mAP metrics
        predictions = detectron2converter(predictions)
        map, _, _ = mean_average_precision(bb_gt, predictions, method='score')
        print('Test mAP for k=' + str(k) + ': ' + str(map))


def task_2():
    print('TODO')

def main():
    task_1_1()
    task_1_2()
    task_2()

main()