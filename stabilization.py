import cv2
from week_4_bm import block_matching
from utils.flow import *
import os
from scipy.signal import medfilt

start = 1
end = 169
prev_frame = None

# Resize frame for computational reasons
w = 480 # 480
h = 270 # 270

frame_dir = "stb_frames_" + str(start) + "_" + str(end) + "_" + str(w) + "_" + str(h) + "/"
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

frames_folder = "datasets/stabilization/seq2/"

direc = 'forward'
blk = 32
bor = 16
met = "template"

def stabilize_frame(frame, estimated_of, w, h, acc_t, method='average'):

    if method == 'average':
        # Average estimated optical flow
        print(np.shape(estimated_of))
        #print(estimated_of)
        avg_estimated_of = - np.array(estimated_of.mean(axis=0).mean(axis=0), dtype=np.float32)
        print(avg_estimated_of)

        acc_t += avg_estimated_of[0:2]
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
        estimated_of = estimated_of.flatten().reshape(h * w, 3)
        np.random.shuffle(estimated_of)
        estimated_of[:,0] = medfilt(estimated_of[:,0], 5)
        estimated_of[:,1] = medfilt(estimated_of[:,1], 5)
        estimated_of[:,2] = medfilt(estimated_of[:,2], 5)

        # Average
        avg_estimated_of = - np.array(estimated_of.mean(axis=0), dtype=np.float32)
        print(avg_estimated_of)
        acc_t += avg_estimated_of[0:2]
        H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
        frame_stabilized = cv2.warpAffine(frame, H, (w, h))

        return frame_stabilized, acc_t


acc_t = np.zeros(2)
acc_total = []

for i in range(start, end):
    # if i == 3 or i == 4:
    dir_frame = frames_folder + "frame_" + str(str(i).zfill(4)) + ".jpg"
    print(dir_frame)

    frame_orig = cv2.imread(dir_frame, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame_orig, (w, h), interpolation=cv2.INTER_AREA)
    # cv2.imshow("current frame", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # First frame case
    if i == start:
        frame_stb = frame
    else:
        # Estimate optical flow
        # Img past: prev frame, img future: frame
        estimated_of = block_matching(prev_frame, frame, direc, blk, bor, met)
        # dense_of_plot(estimated_of, prev_frame, "frame_" + str(i))
        # arrow_of_plot(estimated_of, prev_frame, "frame_" + str(i), custom_scale=False)
        print(estimated_of)
        stb_frame, acc_t = stabilize_frame(frame, estimated_of, w, h, acc_t, 'med_average')
        cv2.cvtColor(stb_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_dir + "frame_stb_" + str(i) + ".jpg", stb_frame)
    
    # Update previous frame for the next iteration
    prev_frame = frame
    acc_total.append(acc_t)