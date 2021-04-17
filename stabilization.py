import cv2
from utils.bm import block_matching
from utils.flow import *
import os
from scipy.signal import medfilt
from utils.stabilization import *
start = 32
end = 236 # 188 # 142
prev_frame = None

# Resize frame for computational reasons
w = 600 # 480
h = 350 # 270

frame_dir = "stb_frames_" + str(start) + "_" + str(end) + "_" + str(w) + "_" + str(h) + "/"
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

frames_folder = "datasets/stabilization/seq1/"

direc = 'forward'
blk = 32
bor = 16
met = "SSD"

acc_t = np.zeros(2)
acc_total = []

for i in range(start, end):
    # if i == 3 or i == 4:
    # dir_frame = frames_folder + "frame_" + str(str(i).zfill(4)) + ".jpg"
    dir_frame = frames_folder + str(str(i).zfill(4)) + ".jpg"
    print(dir_frame)

    frame_orig = cv2.imread(dir_frame, cv2.IMREAD_COLOR)
    # cv2.imshow("current frame", frame_orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    frame = cv2.resize(frame_orig, (w, h), interpolation=cv2.INTER_AREA)

    # First frame case
    if i == start:
        frame_stb = frame
    else:
        # Estimate optical flow
        # Img past: prev frame, img future: frame
        estimated_of = block_matching(prev_frame, frame, direc, blk, bor, met)
        # dense_of_plot(estimated_of, prev_frame, "frame_" + str(i))
        # arrow_of_plot(estimated_of, prev_frame, "frame_" + str(i), custom_scale=False)
        stb_frame, acc_t = stabilize_frame(frame, estimated_of, w, h, acc_t, 'average')
        #stb_frame = stb_frame_of(frame, estimated_of, w, h)
        cv2.cvtColor(stb_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_dir + "frame_stb_" + str(i) + ".jpg", stb_frame)
    
    # Update previous frame for the next iteration
    prev_frame = frame
    acc_total.append(acc_t)