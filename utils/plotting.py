import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_detections(detection_list, gt_list=None, show=True, seq='c010'):
    """
    Plots the detections in detection_list in frame indicated
    by the first detection
    Parameters:
        detection_list: [Detection,...]
        gt_list: [Detection,...]
    """
    if len(detection_list) == 0:
        return np.zeros((10,10,3))

    img_path = './datasets/aicity/AICity_data/train/S03/'+seq+'/frames/frame_' + str(detection_list[0].frame+1).zfill(4) + '.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    for det in detection_list:
        cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), color=(0,255,0), thickness=3)
    
    if gt_list is not None:
        for det in gt_list:
            cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), color=(0,0,255), thickness=3)
    
    if show:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
    
    return img


def plot_detections2(img, detection_list, gt_list=None, show=True):
    """
    Plots the detections in detection_list in img
    Parameters:
        detection_list: [Detection,...]
        gt_list: [Detection,...]
    """

    for det in detection_list:
        np.random.seed(int(det.id))
        c = list(np.random.choice(range(int(256)), size=3)) 
        color = (int(c[0]), int(c[1]), int(c[2]))
        cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), color, 3) 
        cv2.putText(img,str(int(det.id)), (int(det.xtl), int(det.ytl)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if gt_list is not None:
        for det in gt_list:
            cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), color=(0,0,255), thickness=5)
    
    if show:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
    
    return img