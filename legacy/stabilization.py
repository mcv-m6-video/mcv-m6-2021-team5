import cv2
import numpy as np
from scipy.signal import medfilt

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def stb_frame_of(frame, estimated_of, w, h):
    mag, ang = cv2.cartToPolar(estimated_of[:,:,0], estimated_of[:,:,1])
    uniques, counts = np.unique(mag, return_counts=True)
    mc_mag = uniques[counts.argmax()]

    uniques, counts = np.unique(ang, return_counts=True)
    mc_ang = uniques[counts.argmax()]

    u, v = pol2cart(mc_mag, mc_ang)

    affine_H = np.float32([[1, 0, -v],[0,1,-u]])

    frame_stabilized = cv2.warpAffine(frame, affine_H, (w, h))
    
    return frame_stabilized

def stabilize_frame(frame, optical_flow, w, h, acc_t, method='average'):

    if method == 'average':
        # Average
        average_optical_flow = - np.array(optical_flow.mean(axis=0).mean(axis=0), dtype=np.float32)
        acc_t += average_optical_flow
        H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
        frame_stabilized = cv2.warpAffine(frame, H, (w, h))

    if method == 'med_average':
        # Median
        optical_flow = optical_flow.flatten().reshape(h * w, 2)
        np.random.shuffle(optical_flow)
        optical_flow[:,0] = medfilt(optical_flow[:,0], 5)
        optical_flow[:,1] = medfilt(optical_flow[:,1], 5)
        # Average
        average_optical_flow = np.array(-optical_flow.mean(axis=0), dtype=np.float32)
        acc_t += average_optical_flow
        H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
        frame_stabilized = cv2.warpAffine(frame, H, (w, h))

    return frame_stabilized, acc_t