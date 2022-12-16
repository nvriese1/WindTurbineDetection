# Wind Turbine Detection via YOLOv7

## Summary
Implementation of transfer learning approach via the Pytorch 
[YOLOv7 object detection architecture](https://github.com/WongKinYiu/yolov7) to detect and rapidly quantify wind turbines in raw satellite imagery.
<br /><br />
**Image**: _Turbine detections shown in a wind-producing region of the Southwest United States._
<img src="https://user-images.githubusercontent.com/99038816/202049025-25310606-16aa-4ecc-be39-44bccf73579d.jpg" width=60% height=60%>

## Table of Contents

- [Overview & Problem Addressed](#overview-&-problem-addressed)
- [Project Organization](#project-organization)
- [Built With](#built-with)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview & Problem Addressed

 &nbsp;&nbsp;&nbsp;&nbsp;Using raw [LANDSAT](https://figshare.com/articles/dataset/Power_Plant_Satellite_Imagery_Dataset/5307364?file=9104302) 
 and [NAIP](https://datagateway.nrcs.usda.gov/GDGHome_DirectDownLoad.aspx) satellite imagery, a wind turbine object detection model was developed
 via a transfer learning approach from the state-of-the-art YOLOv7 architecture for the purpose of automating on-shore U.S. wind turbine count estimations.
<br />&nbsp;&nbsp;&nbsp;&nbsp;Current state-of-the-art databases which monitor wind turbine development in the United States such as the [U.S. Wind Turbine Database](https://eerscmap.usgs.gov/uswtdb/) are capable of exceptional accuracy, but suffer from poor temporal resolution (updated quarterly). This model, when paired with sufficiently recent satellite imagery data, can provide leading estimates of U.S. on-shore wind resources for both foreign and domestic investors, and government officials, providing value especially within regions of ongoing development.
 
 <img src="https://user-images.githubusercontent.com/99038816/202049644-0f09543a-80b0-433e-889b-795b815eaf94.png" width=60% height=60%>

## Project Organization

    ├── LICENSE
    ├── README.md                                                <- Top-level README for developers using this project.
    |
    ├── notebooks                                                <- Notebooks folder.
    │   └── 1.0-TurbineDetection-data-wrangling.ipynb            <- Imagery data wrangling & EDA notebook.
    │   └── 2.0-TurbineDetection-traning-evaluation.ipynb        <- Google Colab model training notebook.
    │   └── 3.0-TurbineDetection-inference-visualization.ipynb   <- Model inference and evaluation notebook.
    |   └── data.yaml                                            <- YAML file for custom model.
    |   └── detect.py  
    |   └── export.py
    |   └── hubconf.py
    |   └── inference.py
    |   └── test.py
    |   └── train.py
    |   └── requirements.txt                                     <- Required dependencies. 
    |   └── models                                               <- Folder containing additional models and experimental features.
    |   └── utils                                                <- Folder containing additional functions.
    |
    ├── data                                                     <- Data and results folder.
    │     └── cleaned                                            <- Cleaned/augmented image data folder.
    |            └── train                                       <- Training data split.
    |                  └── labels
    |                  └── images
    |            └── test                                        <- Test data split.
    |                  └── labels
    |                  └── images
    |            └── valid                                       <- Validation data split.
    |                  └── labels
    |                  └── images
    |    └── raw                                                 <- Raw annotated image data folder.
    |            └── images
    |            └── labels 
    |    └── results                                             <- Folder containing model metrics and inference images.
    │            └── detections                                  <- Folder containing output model inference images.
    │            └── **metrics**                                 <- **Folder containing model performance metrics.**
    |
    ├── reports                                                  <- Generated analysis as PPTX.
    │   └── turbine_detection_presentation.pptx   
    │ 
    ├── src                                                      <- Source code from notebooks developed for this project.
        └── 1.0-TurbineDetection-data-wrangling.py
        └── 2.0-TurbineDetection-traning-evaluation.py
        └── 3.0-TurbineDetection-inference-visualization.py

## Built With

<a><button name="button">`Python`</button></a> <br />
<a><button name="button">`Jupyer Notebook`</button></a> <br />
<a><button name="button">`Google Colab`</button></a> <br />
<a><button name="button">`Pytorch`</button></a> <br />
<a><button name="button">`Scikit-Learn`</button></a> <br />
<a><button name="button">`Pandas`</button></a> <br />    

## Contact

Noah Vriese<br />
Email: noah@datawhirled.com<br />
Github: [nvriese1](https://github.com/nvriese1)<br />
LinkedIn: [noah-vriese](https://www.linkedin.com/in/noah-vriese/)<br />
Facebook: [noah.vriese](https://www.facebook.com/noah.vriese)<br />
Twitter: [@nvriese](https://twitter.com/nvriese)<br />

## Acknowledgements

WongKinYiu: [YOLOv7 implementation](https://github.com/WongKinYiu/yolov7)<br />
Liscense: MIT
