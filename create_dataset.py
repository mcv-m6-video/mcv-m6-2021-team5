import os
import numpy as np
import cv2

path_to_seqs = 'aic19-track1-mtmc-train/train/S01/'

# Loop over different views (cameras)
for camera in os.listdir(path_to_seqs):
    if str(camera) == 'c001':

        frames = []
        dets_in_frame = []
        detections = []
        prev_frame = 0

        print(camera)
        # Get ground truth txt 
        path_to_gt = path_to_seqs + str(camera) + "/gt/gt.txt"
        path_to_vid = path_to_seqs + str(camera) + "/vdo.avi"
        print(path_to_gt)
        with open(path_to_gt) as f:
            lines = f.readlines()
        # Loop over detections
        for i, line in enumerate(lines):
            print(i)
            print(line)

            data = line.split(',')
            num_frame = int(data[0]) 

            if i == 0:
                prev_frame = num_frame

            object_id = int(data[1])

            print("Num frame: " + str(num_frame))
            print("ID: " + str(object_id))

            xtl = float(data[2])
            ytl = float(data[3])
            width = float(data[4])
            height = float(data[5])

            print(xtl)
            print(ytl)
            print(width)
            print(height)
            
            # if (num_frame != prev_frame and i != 0) or i == len(lines) - 1:
            #     frames.append(num_frame)
            #     detections.append(dets_in_frame)
            #     dets_in_frame = []

            if num_frame != prev_frame:
                frames.append(prev_frame)
                detections.append(dets_in_frame)
                dets_in_frame = []
                prev_frame = num_frame

            # if i == len(lines) - 1:
            #     print(line)

        
            dets_in_frame.append([num_frame, object_id, str(camera), xtl, ytl, width, height])
        
        frames.append(prev_frame)
        detections.append(dets_in_frame)
        dets_in_frame = []
        prev_frame = num_frame

        
print(detections)
print(np.shape(detections))
print(np.shape(frames))

print(frames)
print(len(lines))

video_cap = cv2.VideoCapture(path_to_vid)
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(video_n_frames)
success, input_frame = video_cap.read()

for frame_vid in range(0, int(video_n_frames)):
    # print(frame_vid)
    success, read_frame = video_cap.read()
    if not success:
      break
    if frame_vid in frames:
        print(frame_vid)
        idx = frames.index(frame_vid)
        for detection in detections[idx]:
            print(detection)
            # Crop detection
            cropped_det = read_frame[x:y, x:y]
            cv2.imwrite(str(camera) + "_" + detection[1] + ".jpg", cropped_det)
        
# for i, frame in enumerate(frames):
#     print("Detections for frame " + str(frame))
#     print(detections[i])
