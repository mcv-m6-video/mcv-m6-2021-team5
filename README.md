# Video Surveillance for Road Traffic Monitoring - Team 5
## Contributors
- Eloi Bové (eloi.bove@estudiantat.upc.edu)
- Jordi Burgués (jordiburguesm@gmail.com)
- Albert Jiménez (albert.jimenez.tauste@gmail.com)
- Arnau Roche (arnau.roche@gmail.com)

## Week 1
The goals of this week are: 
- Getting familiar with the frameworks and datasets used in this project
- Implementing the evaluation metrics and plots that are necessary for the following weeks

Specifically, the work is divided in 4 tasks:
- **Task 1:** Detection metrics
- **Task 2:** Temporal analysis of the results
- **Task 3:** Quantitative evaluation of Optical Fow
- **Task 4:** Optical Flow plot


## Week 2
The goals of this week are:
 - Implement and discuss gaussian background models, both adaptive and non-adaptive
 - Compare our results with the state of the art
 - Implement a color extension for our models to improve the results

The task division is as follows:
 - **Task1.1:** Gaussian. Implementation
 - **Task1.2:** Gaussian. Discussion
 - **Task2.1:** Adaptive modeling 
 - **Task2.2:** Adaptive vs non-adaptive models
 - **Task3:** Comparison with the state of the art
 - **Task4:** Color sequences

## Week 3
The goals of this week are:
 - Object detection: evaluate inference detection with off-the-shelf models and fine-tune a model with our data and check if there are improvements
 - Object tracking: implement and compare tracking by maximum overlap and with Kalman Filter

The task division is as follows:
 - **Task1.1:** Object detection. Off-the-shelf inference
 - **Task1.2:** Object detection. Fine-tune to our data
 - **Task2.1:** Object tracking. Maximum overlap
 - **Task2.2:** Object tracking. Kalman Filter

Detectron2 is the framework used, and GPU availability is important if you want to use the models.

## Week 4
The goals of this week are:
 - Optical flow computation using block matching
 - Comparison with off the shelf, state of the art optical flow
 - Video stabilization using optical flow
 - Comparison with state of the art video stabilization
 - Object tracking using optical flow

The task division is as follows:
 - **Task1.1:** Optical flow: Optical Flow with Block Matching
 - **Task1.2:** Optical flow: Off-the-shelf Optical Flow
 - **Task2.1:** Video stabilization: Video stabilization with Block Matching
 - **Task2.2:** Video stabilization: Off-the-shelf Stabilization
 - **Task3.1:** Tracking: Object Tracking with Optical Flow

### Task 1.2: Running optical flow with MaskFlowNet
The following colab link (for anyone with access from upc.edu google suite) shows how to run the MaskFlowNet with the two example images from KITTI. This was done in colab due to the ease of implementation and the simplicity of the code. The rest of the optical flow methods for task 1.2 are implemented in our repo.
https://colab.research.google.com/drive/14kBMx_GR0B_pm1ZwxAhcpNCVSRmmUQF2?usp=sharing


## Installation :wrench:
### Frames generation from vdo.avi
In order to preview the different detections in the frames, vdo.avi should be downloaded in `datasets/aicity/AICity_data/train/S03/c010` and splitted in frames using ffmpeg:

`cd datasets/aicity/AICity_data/train/S03/c010`
`ffmpeg -i vdo.avi -f image2 frames/frame_%04d.png`
``

### Dependencies
`pip install xmltodict`

`pip install tqdm`

`pip install scipy`

`pip install sklearn`

`pip install motmetrics`

`pip install scikit-image`

`pip install filterpy`

`pip install lap`

`sudo apt-get install python3-tk`

`pip install scikit-image`

#### PyFlow
`python3 pyflow/setup.py build_ext -i`

  
## Execution  :gear:
The program is executed as follows, for week X:

`python week_X.py`

### Week 4:
For week 4, the tasks are divided in different scripts:
#### Block matching:
`python week_4_bm.py`
#### Optical flow:
`python week_4_flow.py`
#### Stabilization:
`python week_4_stb.py` 

## Results :clipboard:
All the generated figures are shown or saved to the the directory `figures/`

## Known issues
The cv2.findContours opencv function has a different number of parameters for different versions of opencv, if this function throws an error, change to opencv 4.2 or later.