import cv2
import numpy as np
from utils.bm import *
from utils.flow import *
from tqdm import tqdm
import time
import pickle as pkl

f = open("results.txt", "a")

def block_matching(img_past, img_future, estimation_dir, block_size, search_border, method):
    height, width = img_past.shape[:2]
    of = np.zeros((height, width, 3), dtype=float)

    # Define reference/target image
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
            pos = block_matching_block(ref_block, target_area, method)

            # Scale position to image axis
            u = pos[1] - (j - j_start_area)
            v = pos[0] - (i - i_start_area)

            # Save the optical flow (all pixels are considered as valid: last axis = 1)
            if estimation_dir == "backward":
                of[i:i + block_size, j:j + block_size, :] = [-u, -v, 1]
            if estimation_dir == "forward":
                of[i:i + block_size, j:j + block_size, :] = [u, v, 1]

            # # Plots reference block, search area, and block in target image with minimum distance
            # im_target_show = cv2.rectangle(im_target.copy(),(pos[1] + j_start_area, pos[0] + i_start_area), (pos[1] + j_start_area + block_size, pos[0] + i_start_area + block_size), (255,0, 0), 2) # Plot blue block: block with minimum distance
            # im_target_show = cv2.rectangle(im_target_show,(j_start_area, i_start_area), (j_end, i_end), (0,0,255), 2) # Plot red search area
            # im_target_show = cv2.rectangle(im_target_show,(j, i), (j + block_size,  i + block_size), (0,255,0), 2) # Plot green block: center of search area

            # im_ref_show = cv2.rectangle(im_ref.copy(),(j, i), (j + block_size,  i + block_size), (0, 255, 0), 2) # Plot green block: reference block

            # ref_and_target = np.concatenate((im_target_show, im_ref_show), axis = 0)

            # if i/block_size%16 == 0 and j/block_size%16 == 0:
            #     cv2.imshow("reference and target", ref_and_target)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

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
    estimation_dir = ["backward", "forward"]
    block_size = [4, 8, 16, 32, 64]
    search_border = [4, 8, 16, 32, 64, 128] # Up-down-right-left pixels to look away from block
    search_area = (2*search_border + block_size)
    method = ["SSD", "SAD", "MSE", "MAD", "template"]
    results = []

    for direc in estimation_dir:
        for blk in block_size:
            for bor in search_border:
                for met in method:
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

                    results.append([direc, blk, bor, met, end_time - start_time, msen, pepn])
                    print(results, file = f)
                    print(results)

                    with open('results.pkl', 'wb') as handle:
                        pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    task1()









