import cv2
import numpy as np
# Read target image
color1_path = "datasets/others/colored_0/000157_10.png"
im_target = cv2.imread(color1_path, cv2.IMREAD_COLOR)

def compute_dist(ref_b, target_b, method = 'euclidean'):

    if method == 'euclidean':
        distance = np.sqrt(np.sum((target_b - ref_b) ** 2))

    return distance

def block_matching_show(ref_block, target_area, i_start_area, j_start_area, im_ref_show):
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


def block_matching(ref_block, target_area):
    height_ref = ref_block.shape[0]
    width_ref = ref_block.shape[1]
    min_dist = np.zeros(shape=(target_area.shape[0] - height_ref, target_area.shape[1] - width_ref))
    # print(min_dist.shape)
    # Exhaustive search
    for i in range(0, target_area.shape[0] - height_ref):
        for j in range(0, target_area.shape[1] - width_ref):
            target_block = target_area[i: i + height_ref, j: j + width_ref]
            
            dist = compute_dist(ref_block, target_block)

            min_dist[i, j] = dist
    
    arg_min_dist = np.unravel_index(np.argmin(min_dist, axis=None), min_dist.shape)

    return arg_min_dist



