# Video Surveillance for Road Traffic Monitoring - Team 5
## Contributors
- Eloi Bové (eloi.bove@estudiantat.upc.edu)
- Jordi Burgués (jburguesm@gmail.com)
- Albert Jiménez (albert.jimenez.tauste@gmail.com)
- Arnau Roche (arnau.roche@gmail.com)

## Week 1
The goals of this week are: 
- Getting familiar with the frameworks and datasets used in this project
- Implementing the evaluation metrics and plots that are necessary for the following weeks

Specifically, the work is divided in 4 tasks:
- **Task 1:**  Detection metrics
- **Task 2:** Temporal analysis of the results
- **Task 3:** Quantitative evaluation of Optical Fow
- **Task 4:** Optical Flow plot


## Installation :wrench:
### Frames generation from vdo.avi
In order to preview the different detections in the frames, vdo.avi should be downloaded in `datasets/aicity/AICity_data/train/S03/c010` and splitted in frames using ffmpeg:

`cd datasets/aicity/AICity_data/train/S03/c010`
`ffmpeg -i vdo.avi -f image2 frames/frame_%04d.png`

### Dependencies
`pip install xmltodict`
`pip install tqdm`
`pip install scipy`
`pip install sklearn`
  
## Execution  :gear:
The program is executed as follows:

`python week_1.py`

## Results :clipboard:
All the generated figures are shown or saved to the the directory `figures/`



### Installing pytorch + detectron (CPU only)
`pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
  
`pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html`


