from week_4_bm import block_matching
import cv2
import numpy as np
from utils.bm import *
from utils.flow import *
from tqdm import tqdm
import time

f = open("output.txt", "a")

# Read GT optical flow
gt_of = read_of("datasets/flow/gt/flow_noc/000045_10.png")

# Read past and future images
img_past = cv2.imread("datasets/others/colored_0/000045_10.png", cv2.IMREAD_COLOR)
img_future = cv2.imread("datasets/others/colored_0/000045_11.png", cv2.IMREAD_COLOR)

# Compute optical flow
estimation_dir = ["backward", "forward"]
block_size = [4, 8, 16, 32, 64]
search_border = [4, 8, 16, 32, 64, 128] # Up-down-right-left pixels to look away from block
search_area = (2*search_border + block_size)
method = ["SSD", "SAD", "MSE", "MAD", "template"]

direc = "backward"
blk = 4
bor = 4
met = "template"

print("NEW ITERATION: \n\tEstimation direction: ", direc, "\n\tBlock size: ", blk, "\n\tSearch border: ", bor, "\n\tMethod: ", met, file = f)
print("NEW ITERATION: \n\tEstimation direction: ", direc, "\n\tBlock size: ", blk, "\n\tSearch border: ", bor, "\n\tMethod: ", met)
start_time = time.time()
estimated_of = block_matching(img_past, img_future, direc, blk, bor, met)
end_time = time.time()
print("Elapsed time: ", end_time - start_time, file = f)
print("Elapsed time: ", end_time - start_time)

# Compute metrics
filename = "bm_" + str(direc) + "_" + str(blk) + "_" + str(bor) + "_" + str(met)
dense_of_plot(estimated_of, img_past, filename)
arrow_of_plot(estimated_of, img_past, filename, custom_scale=False)
msen, pepn, of_error1 = compute_of_metrics(estimated_of, gt_of)
plot_of_error(of_error1, filename=filename)

print("MSEN: ", msen, file = f)
print("PEPN: ", pepn, file = f)
print("---------------------", file = f)
print("MSEN: ", msen)
print("PEPN: ", pepn)
print("---------------------")