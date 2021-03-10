# Video Surveillance for Road Traffic Monitoring - Team 5
## Contributors
- Eloi Bové (eloi.bove@estudiantat.upc.edu)
- Jordi Burgués (jburguesm@gmail.com)
- Albert Jiménez (albert.jimenez.tauste@gmail.com)
- Arnau Roche (arnauroche@gmail.com)

## Week 1
The goals of this week are: 
- getting familiar with the frameworks and datasets used in this project
- implementing the evaluation metrics and plots that are necessary for the following weeks

Specifically, the work is divided in 4 tasks:
- **Task 1:**  Detection metrics
- **Task 2:** Temporal analysis of the results
- **Task 3:** Quantitative evaluation of Optical Fow
- **Task 4:** Optical Flow plot

### Frames generation from vdo.avi
`ffmpeg -i vdo.avi -f image2 frames/frame_%04d.png`

### Execution
The program is executed as follows:

`python week_1.py`

### Results
All the generated figures are saved to the the directory `figures/`

## Dependencies
`
  pip install xmltodict
`
### Installing pytorch + detectron (CPU only)
`pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
  
`pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html`


