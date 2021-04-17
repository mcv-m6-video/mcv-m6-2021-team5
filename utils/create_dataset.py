import os
import numpy as np
import cv2

seq = ["S01", "S03", "S04"]

for s in seq:
    path_to_seqs = '../datasets/aic19-track1-mtmc-train/train/' + s + '/'
    path_to_crops = "../datasets/cars/" + s + "/"
    os.makedirs(path_to_crops)

    # Loop over different views (cameras)
    for camera in os.listdir(path_to_seqs):
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

            if num_frame != prev_frame:
                frames.append(prev_frame)
                detections.append(dets_in_frame)
                dets_in_frame = []
                prev_frame = num_frame
        
            dets_in_frame.append([num_frame, object_id, str(camera), xtl, ytl, width, height])
        
        frames.append(prev_frame)
        detections.append(dets_in_frame)
        dets_in_frame = []
        prev_frame = num_frame


        video_cap = cv2.VideoCapture(path_to_vid)
        video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(video_n_frames)
        success, input_frame = video_cap.read()
        for frame_vid in range(2, int(video_n_frames)):
            # print(frame_vid)
            success, read_frame = video_cap.read()
            if not success:
                break
            if frame_vid in frames:
                print("FRAME: ", frame_vid)
                idx = frames.index(frame_vid)
                for det in detections[idx]:
                    # Crop detection
                    cropped_det = read_frame[int(det[4]):int(det[4]+det[6]), int(det[3]):int(det[3]+det[5])]
                    # cv2.imshow("hey", cropped_det)
                    # cv2.waitKey(0)
                    cv2.imwrite(path_to_crops + "{:04}".format(int(det[1])) + "_" + str(camera) + "_" + str(frame_vid) + ".jpg", cropped_det)
            