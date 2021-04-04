import cv2
import numpy as np
from utils.bm import *
from utils.flow import *
from tqdm import tqdm
import pdb

gt1_path = "datasets/flow/gt/flow_noc/000045_10.png"
color1A_path = "datasets/others/colored_0/000045_10.png"
color1B_path = "datasets/others/colored_1/000045_10.png"

# gt2_path = "datasets/flow/gt/flow_noc/000157_10.png"


# Read reference image
im_ref = cv2.imread(color1B_path, cv2.IMREAD_COLOR)

# Read target image
im_target = cv2.imread(color1A_path, cv2.IMREAD_COLOR)

# cv2.imshow("ref im", im_ref)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("target im", im_target)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

height, width = im_ref.shape[:2]

print(height)
print(width)

block_size = 26
# Up-down-right-left pixels to look away from block
search_border = 58

height, width = im_ref.shape[:2]

of_field = np.zeros((height, width, 3), dtype=float)

# Iter rows
for i in tqdm(range(0, height - block_size, block_size)):
    # Iter cols
    for j in range(0, width - block_size, block_size):

        i_start_area = max(i - search_border, 0)
        j_start_area = max(j - search_border, 0)

        i_end =  min(i + block_size + search_border, height)
        j_end = min(j + block_size + search_border, width)

        ref_block = im_ref[i: i + block_size, j: j + block_size]
        target_area = im_target[i_start_area: i_end, j_start_area:j_end]

        pos = block_matching(ref_block, target_area)
        # print(pos)

        u = pos[1] - (j - j_start_area)
        v = pos[0] - (i - i_start_area)

        # print(u)
        # print(v)
        
        of_field[i:i + block_size, j:j + block_size, :] = [u, v, 1]

        # print("Block vars")
        # print(i)
        # print(j)

        # print("Target block vars with min dist")
        # print(pos[0])
        # print(pos[1])

        # Plots reference block, search area, and block in target image with minimum distance
        im_target_show = cv2.rectangle(im_target.copy(),(pos[1] + j_start_area, pos[0] + i_start_area), (pos[1] + j_start_area + block_size, pos[0] + i_start_area + block_size), (255,0, 0), 2)
        im_target_show = cv2.rectangle(im_target_show,(j_start_area, i_start_area), (j_end, i_end), (0,0,255), 2)
        im_target_show = cv2.rectangle(im_target_show,(j, i), (j + block_size,  i + block_size), (0,255,0), 2)

        im_ref_show = cv2.rectangle(im_ref.copy(),(j, i), (j + block_size,  i + block_size), (0, 255, 0), 2)

        ref_and_target = np.concatenate((im_target_show, im_ref_show), axis = 0)

        cv2.imshow("reference and target", ref_and_target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Plots for a specific block, its search area and all possible target blocks in target area moving along
        # if i == block_size*4 and j == block_size*4:

        #     # Visualization
        #     im_target_show = cv2.rectangle(im_target.copy(),(j_start_area, i_start_area), (j_end, i_end), (0,0,255), 2)
        #     im_target_show = cv2.rectangle(im_target_show,(j, i), (j + block_size,  i + block_size), (0,255,0), 2)

        #     im_ref_show = cv2.rectangle(im_ref.copy(),(j, i), (j + block_size,  i + block_size), (0, 255, 0), 2)

        #     ref_and_target = np.concatenate((im_target_show, im_ref_show), axis = 0)

        #     cv2.imshow("reference and target", ref_and_target)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        #     pos = block_matching_show (ref_block, target_area, i_start_area, j_start_area, ref_and_target)
        #     print(pos)

        #     u = pos[1] - (j - j_start_area)
        #     v = pos[0] - (i - i_start_area)

        #     print(u)
        #     print(v)
            
        #     of_field[i:i + block_size, j:j + block_size, :] = [u, v, 1]


search_area = (2*search_border + block_size)
print("Search area: " + str(search_area) + "x" + str(search_area))



# Compute metrics
gt1 = read_of(gt1_path)
dense_of_plot(of_field, im_ref, "bm_of_157_10_" + str(block_size) + "_" + str(search_border))
arrow_of_plot(of_field, im_ref, "bm_of_157_10_" + str(block_size) + "_" + str(search_border), custom_scale=False)
msen, pepn, of_error1 = compute_of_metrics(of_field, gt1)
#plot_of_error(of_error1, filename="1")

print(of_field[200,200,0])
print(gt[200,200,0])

print(msen)
print(pepn)

pdb.set_trace()

