import cv2
import numpy as np
from utils.bm import *
from utils.flow import *
from tqdm import tqdm

def block_matching(img_past, img_future, estimation_dir, block_size, search_border):
    height, width = img_past.shape[:2]
    of = np.zeros((height, width, 3), dtype=float)

    # Define reference/target image according to the desired method
    if estimation_dir == "backward":
        im_ref = img_future
        im_target = img_past
    elif estimation_dir == "forward":
        im_ref = img_past
        im_target = img_future

    # Iter rows
    for i in tqdm(range(0, height - block_size, block_size)):
        # Iter cols
        for j in range(0, width - block_size, block_size):

            # Crop reference block and target area to search
            i_start_area = max(i - search_border, 0)
            j_start_area = max(j - search_border, 0)

            i_end =  min(i + block_size + search_border, height)
            j_end = min(j + block_size + search_border, width)

            ref_block = im_ref[i: i + block_size, j: j + block_size]
            target_area = im_target[i_start_area: i_end, j_start_area:j_end]

            # Obtain the position of the block with lower distance
            pos = block_matching_block(ref_block, target_area)

            # Scale position to image axis
            u = pos[1] - (j - j_start_area)
            v = pos[0] - (i - i_start_area)

            # Save the optical flow (all pixels are considered as valid: last axis = 1)
            of[i:i + block_size, j:j + block_size, :] = [u, v, 1]

            # Plots reference block, search area, and block in target image with minimum distance
            im_target_show = cv2.rectangle(im_target.copy(),(pos[1] + j_start_area, pos[0] + i_start_area), (pos[1] + j_start_area + block_size, pos[0] + i_start_area + block_size), (255,0, 0), 2) # Plot blue block: block with minimum distance
            im_target_show = cv2.rectangle(im_target_show,(j_start_area, i_start_area), (j_end, i_end), (0,0,255), 2) # Plot red search area
            im_target_show = cv2.rectangle(im_target_show,(j, i), (j + block_size,  i + block_size), (0,255,0), 2) # Plot green block: center of search area

            im_ref_show = cv2.rectangle(im_ref.copy(),(j, i), (j + block_size,  i + block_size), (0, 255, 0), 2) # Plot green block: reference block

            ref_and_target = np.concatenate((im_target_show, im_ref_show), axis = 0)

            if i/block_size%16 == 0 and j/block_size%16 == 0:
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

            #     pos = block_matching_block_show (ref_block, target_area, i_start_area, j_start_area, ref_and_target)
            #     print(pos)

            #     u = pos[1] - (j - j_start_area)
            #     v = pos[0] - (i - i_start_area)

            #     print(u)
            #     print(v)
                
            #     of[i:i + block_size, j:j + block_size, :] = [u, v, 1]
    return of


def task1():
    # Estimate optical flow with block matching

     # Read GT optical flow
    gt_of = read_of("datasets/flow/gt/flow_noc/000045_10.png")

    # Read past and future images
    img_past = cv2.imread("datasets/others/colored_0/000045_10.png", cv2.IMREAD_COLOR)
    img_future = cv2.imread("datasets/others/colored_0/000045_11.png", cv2.IMREAD_COLOR)

    # Compute optical flow
    estimation_dir = ["backward"]
    block_size = 16
    search_border = 32 # Up-down-right-left pixels to look away from block
    search_area = (2*search_border + block_size)
    print("Search area: " + str(search_area) + "x" + str(search_area))

    estimated_of = block_matching(img_past, img_future, estimation_dir[0], block_size, search_border)

    # Compute metrics
    dense_of_plot(estimated_of, img_past, "bm_of_157_10_" + str(block_size) + "_" + str(search_border))
    arrow_of_plot(estimated_of, img_past, "bm_of_157_10_" + str(block_size) + "_" + str(search_border), custom_scale=False)
    msen, pepn, of_error1 = compute_of_metrics(estimated_of, gt_of)
    plot_of_error(of_error1, filename="bm")

    print("MSEN: ", msen)
    print("PEPN: ", pepn)


if __name__ == '__main__':
    task1()









