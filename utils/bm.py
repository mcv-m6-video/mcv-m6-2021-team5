import cv2
import numpy as np
from skimage.feature import match_template
from tqdm import tqdm

# Read target image
color1_path = "datasets/others/colored_0/000157_10.png"
im_target = cv2.imread(color1_path, cv2.IMREAD_COLOR)

def block_matching(img_past, img_future, estimation_dir, block_size, search_border, method):
    height, width = img_past.shape[:2]
    of = np.zeros((height, width, 2), dtype=float)

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
                of[i:i + block_size, j:j + block_size, :] = [-u, -v]
            if estimation_dir == "forward":
                of[i:i + block_size, j:j + block_size, :] = [u, v]
    return of


def compute_dist(ref_b, target_b, method = "SSD"):

    if method == "SSD":
        distance = np.sum((target_b - ref_b) ** 2)
    if method == "SAD":
        distance = np.sum(np.abs(target_b - ref_b))
    if method == 'MSE':
        distance = np.mean((target_b - ref_b) ** 2)
    if method == 'MAD':
        distance = np.mean(np.abs(target_b - ref_b))
    return distance

def block_matching_block_show(ref_block, target_area, i_start_area, j_start_area, im_ref_show):
    height_ref = ref_block.shape[0]
    width_ref = ref_block.shape[1]
    min_dist = np.zeros(shape=(target_area.shape[0] - height_ref, target_area.shape[1] - width_ref))
    
    # Exhaustive search
    for i in range(0, target_area.shape[0] - height_ref):
        for j in range(0, target_area.shape[1] - width_ref):
            target_block = target_area[i: i + height_ref, j: j + width_ref]
            
            dist = compute_dist(ref_block, target_block)

            min_dist[i, j] = dist
            

            if j%31 == 0 and i%31 == 0: 
                # Target block
                im_to_show = cv2.rectangle(im_ref_show.copy(), (j + j_start_area, i + i_start_area), (j + j_start_area + width_ref,  i + i_start_area + height_ref), (255, 0, 0), 2)

                cv2.imshow("reference and target", im_to_show)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    arg_min_dist = np.unravel_index(np.argmin(min_dist, axis=None), min_dist.shape)

    return arg_min_dist


def block_matching_block(ref_block, target_area, method):
    height_ref = ref_block.shape[0]
    width_ref = ref_block.shape[1]
    min_dist = np.zeros(shape=(target_area.shape[0] - height_ref, target_area.shape[1] - width_ref))

    # Exhaustive search
    if method == "SSD" or method == "SAD" or method == "MSE" or method == "MAD":
        for i in range(0, target_area.shape[0] - height_ref):
            for j in range(0, target_area.shape[1] - width_ref):
                target_block = target_area[i: i + height_ref, j: j + width_ref]
                
                dist = compute_dist(ref_block, target_block, method)

                min_dist[i, j] = dist
        
        arg_min_dist = np.unravel_index(np.argmin(min_dist, axis=None), min_dist.shape)

    # Match template
    if method == "template":
        corr = match_template(target_area, ref_block)
        arg_min_dist = np.unravel_index(np.argmin(corr, axis=None), corr.shape)

    return arg_min_dist



