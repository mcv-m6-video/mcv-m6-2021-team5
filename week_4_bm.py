import cv2
import numpy as np
from utils.bm import *

color1_path = "datasets/others/colored_0/000157_10.png"
color2_path = "datasets/others/colored_1/000157_10.png"


# Read reference image
im_ref = cv2.imread(color2_path, cv2.IMREAD_COLOR)

# Read target image
im_target = cv2.imread(color1_path, cv2.IMREAD_COLOR)

# cv2.imshow("ref im", im_ref)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("target im", im_target)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

height, width = im_ref.shape[:2]

print(height)
print(width)

block_size = 64
# Up-down-right-left pixels to look away from block
search_border = 4

height, width = im_ref.shape[:2]

of_field = np.zeros((height, width, 2), dtype=float)

# Iter rows
for i in range(0, height - block_size, block_size):
    # Iter cols
    for j in range(0, width - block_size, block_size):

        i_start = max(i - search_border, 0)
        j_start = max(j - search_border, 0)

        i_end =  min(i + block_size + search_border, height)
        j_end = min(j + block_size + search_border, width)

        ref_block = im_ref[i: i + block_size, j: j + block_size]
        target_area = im_target[i_start: i_end, j_start:j_end]

        pos = block_matching(ref_block, target_area)
        # pos = block_matching_show(ref_block, target_area, i, j, block_size)

        # Visualization
        # im_target_show = cv2.rectangle(im_target.copy(),(j_start, i_start), (j_end, i_end), (0,255,0), 2)
        # im_target_show = cv2.rectangle(im_target_show,(j, i), (j + block_size,  i + block_size), (255,0,0), 2)

        # im_ref_show = cv2.rectangle(im_ref.copy(),(j, i), (j + block_size,  i + block_size), (255,0,0), 2)

        # ref_and_target = np.concatenate((im_target_show, im_ref_show), axis = 0)

        # cv2.imshow("reference and target", ref_and_target)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(pos)

        u = pos[1] - (j - j_start)
        v = pos[0] - (i - i_start)

        print(u)
        print(v)
        
        of_field[i:i + block_size, j:j + block_size, :] = [u, v]

search_area = (2*search_border + block_size)
print("Search area: " + str(search_area) + "x" + str(search_area))

print(of_field)


