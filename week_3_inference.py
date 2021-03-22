import torch, torchvision
import detectron2
import numpy as np
import os, cv2, random
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.plotting import plot_detections

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2_tools.io import detectronReader
from evaluation.ap import mean_average_precision
from utils.reader import AnnotationReader


xmlfile = "datasets/aicity/ai_challenge_s03_c010-full_annotation.xml" 

def task_1_1():

    start = 535
    end = 2141

    models = ["COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
              "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
              "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]

    # Read GT in our format for evaluation
    gt_reader = AnnotationReader(xmlfile)
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])

    # Get GT for evaluation
    bb_gt = []
    for frame in range(start, end):
        boxes = []
        for box in gt[frame]:
            boxes.append(box)
        bb_gt.append(boxes)

    # Read the data for benchmarking
    reader = detectronReader(xmlfile)

    # Register the datasets
    for d in ['train', 'val']:
        DatasetCatalog.register("AICity_"+d, lambda d=d: reader.get_dict_from_xml(d[0:-1],K=k))
        MetadataCatalog.get("AICity_"+d).set(thing_classes=['car'])
    aicity_metadata = MetadataCatalog.get("AICity_train")

    # Iterate for each model
    for model in models:
        # Create model for inference
        model_name = (model.split('/')[1]).split('.')[0]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        predictor = DefaultPredictor(cfg)

        # Inference on the test dataset
        val_dict = reader.get_dict_from_xml('val')

        # Predict
        predictions = []
        for d in tqdm(val_dict):    
            im = cv2.imread(d["file_name"])
            outputs = predictor(im) 
            predictions.append(outputs)

        # Compute mAP metrics
        predictions = detectron2converter(predictions)
        map, _, _ = mean_average_precision(bb_gt, predictions, method='score')
        print('Validation mAP for '+model_name+': ' + str(map))

        

def task_1_2():
    """
    This function reads the data and fine tunes an existing detectron2 model
    """
    # Read GT in our format for evaluation
    gt_reader = AnnotationReader(xmlfile)
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])

    # K=4 cross validation
    for k in range(0,4):

        if k == 0:

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
            cfg.SOLVER.MAX_ITER = 300  # TODO: Just for testing
            cfg.SOLVER.STEPS = []        
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 


            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(cfg) 
            trainer.resume_or_load(resume=False)
            trainer.train()

            # Inference on the test dataset
            val_dict = reader.get_dict_from_xml('val', K=k)
            
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
            predictor = DefaultPredictor(cfg)
            
            # Predict
            predictions = []
            for d in tqdm(val_dict):    
                im = cv2.imread(d["file_name"])
                outputs = predictor(im) 
                predictions.append(outputs)

            # Get GT for evaluation
            _, range_val = reader.get_range_for_k(k)
            range_val_number = np.shape(range_val)[0] # Get total number of frames for validation
            bb_gt = []
            for frame in range(range_val_number):
                boxes = []
                for box in gt[frame]:
                    boxes.append(box)
                bb_gt.append(boxes)

            # Compute mAP metrics
            predictions = detectron2converter(predictions)
            map, _, _ = mean_average_precision(bb_gt, predictions, method='score')
            print('Validation mAP for k=' + str(k) + ': ' + str(map))
        

def dummy_head():
    # Read GT in our format for evaluation
    gt_reader = AnnotationReader(xmlfile)
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])

    # Init dataset reader
    reader = detectronReader(xmlfile)

    cfg = get_cfg()
    cfg.OUTPUT_DIR = 'detectron2_models/faster_rcnn_X_101_32x8d_FPN_3x_KV_0_secure'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("AICity_train"+str(0),)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 300  # TODO: Just for testing
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    # Inference on the test dataset
    val_dict = reader.get_dict_from_xml('val', K=0)
    
    cfg.MODEL.WEIGHTS = "detectron2_models/faster_rcnn_X_101_32x8d_FPN_3x_KV_0_secure/model_final.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    # Predict
    predictions = []
    for d in tqdm(val_dict):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im) 
        predictions.append(outputs)

    # Get GT for evaluation
    _, range_val = reader.get_range_for_k(0)
    range_val_number = np.shape(range_val)[0] # Get total number of frames for validation
    print(range_val_number)
    bb_gt = []
    for frame in range(536,):
        boxes = []
        for box in gt[frame]:
            boxes.append(box)
        bb_gt.append(boxes)

    # Compute mAP metrics
    predictions = reader.detectron2converter(predictions)
    map, _, _ = mean_average_precision(bb_gt, predictions, method='score')
    print('Validation mAP for k=' + str(0) + ': ' + str(map))

    im_det = plot_detections(predictions[1], bb_gt[1], show=False)
    cv2.imwrite("detections_1.png", im_det)
def task_2():
    print('TODO')

def main():
    dummy_head()
    # task_1_1()
    # task_1_2()
    #task_2()

main()