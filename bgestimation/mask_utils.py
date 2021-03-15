import cv2
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def denoise_mask(mask, method):
    if method==1:
        mask = cv2.medianBlur(mask, 7)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

        # Flood fill
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        # Erode
        mask = cv2.erode(mask, kernel1, iterations=1)
        # Dilate
        mask = cv2.dilate(mask, kernel2, iterations=1)
        return (mask * 255).astype(np.uint8)

    if method==2:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    if method==3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (140, 140)))
        return mask
    elif method==4:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        return mask