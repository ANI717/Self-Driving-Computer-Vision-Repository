
<p align="center">
  <h1 align="center">Computer Vision & Deep Learning Repository</h1>
</p>

<p align="justify">
Computer Vision & Deep Learning Repository for Autonomous Driving of Jetson Nano/Raspberry Pi controlled Miniature Robot Cars. The Deep Learning models are trained to mimic robot car driving controlled by gamepad.
</p>


## Quick Links
[Robotics Workspace](https://github.com/ANI717/ANI717_Robotics)<br/>


## Colaborators
[Computer Fusion Laboratory (CFL) - Temple University College of Engineering](https://sites.temple.edu/cflab/people/)
* [Animesh Bala Ani](https://www.linkedin.com/in/ani717/) (Software Development)<br/>
* [Michael Nghe](https://sites.temple.edu/cflab/people/) (Data Collection)<br/>
* [Dr. Li Bai](https://engineering.temple.edu/about/faculty-staff/li-bai-lbai) (Academic Advisor)<br/>


## Directory Tree (Important Scripts and Database)
```
Self Driving CV Repository
    ├── data process
    │   ├── output
    │   ├── settings
    │   ├── batch_rename.py
    │   ├── create_dataset.py
    │   ├── create_final_dataset.py
    │   ├── refine_dataset.py
    │   └── visually_refine_dataset.py
    ├── dataset
    │   ├── images
    │   └── lists
    │       ├── cross-validation
    │       ├── debug
    │       └── random
    │           ├── train.csv
    │           ├── val.csv
    │           └── test.csv
    ├── deep learning
    │   ├── train
    │   │   ├── checkpoints
    │   │   ├── output
    │   │   ├── config.py
    │   │   ├── utils.py
    │   │   ├── dataset.py
    │   │   ├── model.py
    │   │   └── train.py
    │   └── test
    │   │   ├── test.py
    │   │   └── test_onnx.py
    ├── results
    └── visualization
        ├── prediction_visualization.py
        └── prediction_video.py
```


## Demonstration
<img src="https://github.com/ANI717/ani717_gif_repository/blob/main/temple-race-car-deeplearning/prediction-1.gif" alt="prediction" class="inline"/><br/>
