import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def read_of(flow_path):

    flow_raw = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # Transform data (DevKit Stereo Flow - KITTI)
    flow_u = (flow_raw[:,:,2] - 2**15) / 64.0
    flow_v = (flow_raw[:,:,1] - 2**15) / 64.0
    flow_valid = flow_raw[:,:,0] == 1

    # Set to 0 the points where the flow is not valid
    flow_u[~flow_valid] = 0
    flow_v[~flow_valid] = 0

    # Reorder channels
    return np.stack((flow_u, flow_v, flow_valid), axis=2)

def compute_of_metrics(flow, gt):
    # Binary mask to discard non-occluded areas
    #non_occluded_areas = gt[:,:,2] != 0

    # Only for the first 2 channels
    square_error_matrix = (flow[:,:,0:2] - gt[:,:,0:2]) ** 2
    square_error_matrix_valid = square_error_matrix*np.stack((gt[:,:,2],gt[:,:,2]),axis=2)
    #square_error_matrix_valid = square_error_matrix[non_occluded_areas]

    #non_occluded_pixels = np.shape(square_error_matrix_valid)[0]
    non_occluded_pixels = np.sum(gt[:,:,2] != 0)

    # Compute MSEN
    pixel_error_matrix = np.sqrt(np.sum(square_error_matrix_valid, axis= 2)) # Pixel error for both u and v
    msen = (1/non_occluded_pixels) * np.sum(pixel_error_matrix) # Average error for all non-occluded pixels
    
    # Compute PEPN
    erroneous_pixels = np.sum(pixel_error_matrix > 3)
    pepn = erroneous_pixels/non_occluded_pixels
    
    return msen, pepn, pixel_error_matrix

def dense_of_plot(flow, color_img, filename):

    #Calculate OF magnitude and angles
    flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    flow_direction = np.arctan2(flow[:,:,1], flow[:,:,0]) + np.pi

    #Clip the highest magnitude values according to the 0.95 quantile
    clip_th = np.quantile(flow_magnitude,0.95)
    flow_magnitude = np.clip(flow_magnitude,0,clip_th) 

    #Scale the magnitude so that it takes values within [0,255]
    flow_scaled_magnitude = (flow_magnitude/np.max(flow_magnitude))*255

    #Generate a visualization of the OF in the HSV space
    flow_hsv = np.zeros(flow.shape, dtype=np.uint8)
    flow_hsv[:,:,0] = flow_direction/(2*np.pi)*179
    flow_hsv[:,:,1] = flow_scaled_magnitude
    flow_hsv[:,:,2] = 255
    flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

    #Write images
    cv2.imwrite("figures/magnitude_" + filename + ".png", flow_scaled_magnitude)
    cv2.imwrite("figures/dense_" + filename + ".png", flow_bgr)

    #Combine the OF visualization and the original color image
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    flow_bgr[np.where((flow_bgr==[255,255,255]).all(axis=2))] = [0,0,0]
    flow_on_image = cv2.addWeighted(color_img, 1,flow_bgr, 2, 0)

    #Write image
    cv2.imwrite("figures/dense_on_img_" + filename + ".png", flow_on_image)


def read_img(img):
    return cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

def plot_of_error(of_error, filename):
    plt.figure()
    plt.imshow(of_error)
    plt.axis('off')
    plt.savefig("figures/error_" + filename + ".png", bbox_inches='tight', pad_inches=0, dpi=250)

def arrow_of_plot(flow, img, filename, custom_scale=True): 
    height = np.shape(flow[:,:,0])[0]
    width = np.shape(flow[:,:,0])[1]

    step = 15
    X, Y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))

    U = flow[::step,::step,0]
    V = flow[::step,::step,1]
    M = np.hypot(U, V)
    
    # Plot
    plt.figure()
    plt.axis('off')

    if custom_scale:
        scale = 10*(U.max() + V.max())
        Q = plt.quiver(X, Y, U, V, M, color='red', angles='xy', scale=scale)
    else:
        Q = plt.quiver(X, Y, U, V, M, color='red', angles='xy')
        
    plt.imshow(img)
    plt.savefig("figures/arrows_" + filename + ".png", bbox_inches='tight', pad_inches=0, dpi=250)

    #Clip the highest magnitude values according to the 0.95 quantile
    clip_th = np.quantile(M,0.95)
    U[M>clip_th] = U[M>clip_th]*0.3
    V[M>clip_th] = V[M>clip_th]*0.3
    M = np.hypot(U, V)
    
    # Plot
    plt.figure()
    plt.axis('off')

    if custom_scale:
        scale = 10*(U.max() + V.max())
        Q = plt.quiver(X, Y, U, V, M, color='red', angles='xy', scale=scale)
    else:
        Q = plt.quiver(X, Y, U, V, M, color='red', angles='xy')

    plt.imshow(img)
    plt.savefig("figures/arrows_clipped_" + filename + ".png", bbox_inches='tight', pad_inches=0, dpi=250)

def hist_of_error(of_error, mask=None, filename=None):
    h, w = np.shape(of_error)
    if mask is not None:
        of_error = of_error[np.nonzero(mask)]

    plt.figure()
    plt.hist(of_error.ravel(), bins=50)
    plt.savefig("figures/hist_" + filename + ".png")
