import cv2
import numpy as np
# Read target image
color1_path = "datasets/others/colored_0/000157_10.png"
im_target = cv2.imread(color1_path, cv2.IMREAD_COLOR)

def block_matching(ref_block, target_area):
    height_ref = ref_block.shape[0]
    width_ref = ref_block.shape[1]
    min_dist = np.zeros(shape=(target_area.shape[0] - height_ref, target_area.shape[1] - width_ref))
    # print(min_dist.shape)
    # Exhaustive search
    for i in range(0, target_area.shape[0] - height_ref):
        for j in range(0, target_area.shape[1] - width_ref):
            target_block = target_area[i: i + height_ref, j: j + width_ref]

            # Target block
            cv2.rectangle(im_target,(j, i), (j + width_ref,  i + height_ref), (255,0,0), 2)
            
            dist = np.sqrt(np.sum((target_block - ref_block) ** 2))

            min_dist[i, j] = dist
            # cv2.imshow("reference and target", im_target)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    
    arg_min_dist = np.unravel_index(np.argmin(min_dist, axis=None), min_dist.shape)

    return arg_min_dist



