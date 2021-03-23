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
#from tracking.tracking import track_max_overlap, track_kalman
import pickle as pkl


xmlfile = "datasets/aicity/ai_challenge_s03_c010-full_annotation.xml" 
maps = []

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
        predictions = reader.detectron2converter(predictions, coco=True)
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
        cfg.SOLVER.MAX_ITER = 2  # TODO: Just for testing
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
        predictions = detectron2converter(predictions, coco=False)
        map, _, _ = mean_average_precision(bb_gt, predictions, method='score')
        print('Validation mAP for k=' + str(k) + ': ' + str(map))
        
def task_1_1_bis():
    k = 0
    models = ["COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
    for iteration in (6, 7, 8, 9):
        for mod in models:
            compute(iteration = iteration, output_dir = None, k=k, train = False, validate = True, plot=True, model_name = mod, coco=True)
        #compute(output_dir = None, k=k, train = False, validate = True, plot=True, model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", coco=True)


def task_1_2_bis():
    #for iteration in (200, 400, 600):
    k = 4
    for iteration in (4, 7):
        output_dir = 'detectron2_models/faster_rcnn_X_101_32x8d_FPN_3x_KV_' + str(k) + "_" + str(iteration)
        compute(iteration = iteration, output_dir = output_dir, k=k, train = True, validate = True, plot = True, model_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    print(maps)

    # iteration = 999
    # for k in range(0,5):
    #     output_dir = 'detectron2_models/faster_rcnn_X_101_32x8d_FPN_3x_KV_' + str(k) + "_" + str(iteration)
    #     compute(iteration = iteration, output_dir = output_dir, k=k, train = True, validate = True, plot = True, model_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    # print(maps)

def compute(iteration, output_dir, k, train, validate, plot = False, model_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml", coco=False):

    # Init dataset reader
    reader = detectronReader(xmlfile)

    # Register the datasets for this iteration
    print(iteration)
    for d in ['train'+str(k)+str(iteration), 'val'+str(k)+str(iteration)]:
        print(d)
        print("NAMEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print(d[0:-4])
        DatasetCatalog.register("AICity_"+d, lambda d=d: reader.get_dict_from_xml(d[0:-3],K=k))
        MetadataCatalog.get("AICity_"+d).set(thing_classes=['car'])
    aicity_metadata = MetadataCatalog.get("AICity_"+d)

    # Read the dataset from the xml file
    train_dict = reader.get_dict_from_xml('train',K=k)

    # # Test if data is properly read
    # for d in train_dict[0:1]:
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=aicity_metadata, scale=0.5)
    #     out = visualizer.draw_dataset_dict(d)
    #     plt.imshow(out.get_image()[:, :, ::-1])
    #     plt.show()

    # Config file
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    
    # We have trained a model, we need to retrieve configuration
    if output_dir != None:
        cfg.DATASETS.TRAIN = ("AICity_train"+str(k)+str(iteration),)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  
        cfg.SOLVER.MAX_ITER = iteration  # TODO: Just for testing
        cfg.SOLVER.STEPS = []        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    if train:
        cfg.OUTPUT_DIR = output_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()

    if validate:
        # Inference on the test dataset
        val_dict = reader.get_dict_from_xml('val', K=k)

        if output_dir != None:
            cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")  # path to the model we just trained
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        # Predict
        predictions = []
        for d in tqdm(val_dict):    
            im = cv2.imread(d["file_name"])
            outputs = predictor(im) 
            predictions.append(outputs)

        # Read GT in our format for evaluation
        gt_reader = AnnotationReader(xmlfile)
        gt = gt_reader.get_bboxes_per_frame(classes=['car'])

        # Get GT for evaluation
        # _, range_val = reader.get_range_for_k(k)
        # range_val_number = np.shape(range_val)[0] # Get total number of frames for validation
        bb_gt = []
        
        print(reader.range_val)
        for frame_num in reader.range_val:
            boxes = []
            for box in gt[frame_num]:
                boxes.append(box)
            bb_gt.append(boxes)

        """
        img_path = './datasets/aicity/AICity_data/train/S03/c010/frames/frame_' + str(reader.range_val[0]).zfill(4) + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(predictions[0]["instances"].to("cpu"))
        cv2.imwrite("dets.png", out.get_image()[:, :, ::-1])
        #cv2_imshow(out.get_image()[:, :, ::-1])
        """

        mod_name = (model_name.split('/')[1]).split('.')[0]

        # Save Detectron detections
        with open(mod_name + '_' + str(k) + '_' + str(iteration) + '_detectron.pkl', 'wb') as handle:
            pkl.dump(predictions, handle, protocol=pkl.HIGHEST_PROTOCOL)

        # Compute mAP metrics
        predictions = reader.detectron2converter(predictions, coco=coco)
        map, _, _ = mean_average_precision(bb_gt, predictions, method='score')
        print('Validation mAP for model ' + mod_name + ' and k=' + str(k) + " iteration = " + str(iteration) + ': ' + str(map))
        maps.append(map)

        # Save converted detections
        with open(mod_name + '_' + str(k) + '_' + str(iteration) + '_ours.pkl', 'wb') as handle:
            pkl.dump(predictions, handle, protocol=pkl.HIGHEST_PROTOCOL)

        # Save Detectron detections
        with open(mod_name + '_' + str(k) + '_' + str(iteration) + '_map.pkl', 'wb') as handle:
            pkl.dump(map, handle, protocol=pkl.HIGHEST_PROTOCOL)

    if plot:
        im_det = plot_detections(predictions[0], bb_gt[0], show=False)
        cv2.imwrite("detections_0_" + str(k) + "_" + str(iteration) + ".png", im_det)

        im_det = plot_detections(predictions[9], bb_gt[9], show=False)
        cv2.imwrite("detections_9_" + str(k) + "_" + str(iteration) + ".png", im_det)

def task_2(model='rcnn',method='overlap'):

    # Load GT
    # Read GT in our format for evaluation
    gt_reader = AnnotationReader(xmlfile)
    gt = gt_reader.get_bboxes_per_frame(classes=['car'])

    # Get GT for evaluation
    bb_gt = []
    start, end = 535, 2141
    for frame in range(start, end):
        boxes = []
        for box in gt[frame]:
            boxes.append(box)
        bb_gt.append(boxes)

    if method == 'overlap':
        track_max_overlap(bb_gt, bb_gt)
    elif method == 'kalman':
        track_kalman(bb_gt, bb_gt)
    else:
        print('Invalid tracking method: overlap or kalman')
    

def main():
    # task_1_1()
    # task_1_2()
    task_1_1_bis()
    task_1_2_bis()
    # task_2(method='overlap')
    # task_2(method='kalman')

main()