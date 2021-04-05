import cv2
import numpy as np
from scipy.signal import medfilt

def stabilize_frame(frame, estimated_of, w, h, acc_t, method='average'):

    if method == 'average':
        print("USING AVERAGE")
        # Average estimated optical flow
        print(np.shape(estimated_of))
        #print(estimated_of)
        avg_estimated_of = -np.array(estimated_of.mean(axis=0).mean(axis=0), dtype=np.float32)
        print(avg_estimated_of)

        acc_t += avg_estimated_of
        print(acc_t)
        H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
        print(H)
        frame_stabilized = cv2.warpAffine(frame, H, (w, h))
        print(frame_stabilized)

        # orig_stb = np.concatenate((frame, frame_stabilized), axis = 0)
        # cv2.imshow("current frame", orig_stb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    elif method == 'med_average':

        # Median
        estimated_of = estimated_of.flatten().reshape(h * w, 2)
        # np.random.shuffle(estimated_of)
        estimated_of[:,0] = medfilt(estimated_of[:,0], 5)
        estimated_of[:,1] = medfilt(estimated_of[:,1], 5)

        # Average
        avg_estimated_of = -np.array(estimated_of.mean(axis=0), dtype=np.float32)
        print(avg_estimated_of)
        acc_t += avg_estimated_of
        H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
        frame_stabilized = cv2.warpAffine(frame, H, (w, h))

        # cv2.imshow("current frame", orig_stb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return frame_stabilized, acc_t


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