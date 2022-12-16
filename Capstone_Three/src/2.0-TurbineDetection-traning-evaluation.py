#!/usr/bin/env python
# coding: utf-8

# # **Capstone Project 3: Custom Turbine Detection using YOLOv7 Architecture**
# 
# ## *Notebook 2/3: Model Pre-Processing, Model Development, Training*
# 
# This notebook details the steps required to implement a custom object detection model using the YOLOv7 (You Only Look Once) framework (built in Pytorch). The following example shows a transfer learning implementation begining from the pre-trained YOLOv7 model. A custom aerial imagery dataset acquired from [Kaggle - Wind Turbine Detection](https://www.kaggle.com/datasets/saurabhshahane/wind-turbine-obj-detection) was annotated, augmented and exported via Roboflow API prior to import for custom object detection training. 
# 
# Acknowlegements to WongKinYiu for creating the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) which was utilized significantly in this work.
# 
# ### *Steps Covered in this Notebook*
# 
# To train our object detection model, the following steps are required:
# 
# 1. Setup file storage through Google Drive
# 2. Import custom dataset (API or YOLOv7 formatted .zip file)
# 3. Custom model training
# 4. Evaluate model performance & run inference on test set images
# 5. Export model weights and metrics
# 
# 
# ### *Preparing a Custom Dataset*
# 
# Prior to model development and training, a custom dataset should be acquired. In this example, an imagery dataset containing aerial photos of satellite images was downloaded from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/wind-turbine-obj-detection). Included in the raw dataset are YOLOv3 bounding box .txt annotations for 1,742 (608px x 608px) .jpg images, 1,340 of which are non-null (containing at least one turbine).

# ### *Note: Google Colab users*
# If executing this notebook in __Google Colab__, remember to begin by starting a new runtime. It is recommended to use a GPU (freely offered) to expedite model training, if required.
# ___

# # 1.0 Setup File Storage (Google Drive)
# 
# Google Drive will be used as the primary method of file storage in this notebook. To begin, import the Google Colab-Drive storage utility and mount your Google Drive in the workspace. When prompted, allow access to make changes in your Drive.

# In[ ]:


# import Google Drive storage utility API
from google.colab import drive

# mount drive to Google Colab workspace
drive.mount('/content/gdrive', force_remount=True)


# Ensure **'gdrive/'** and **'sample_data/'** are present in the current working directory. If so, the Google Drive has been mounted successfully.

# In[ ]:


# check the current working directory to ensure successful mount
get_ipython().run_line_magic('ls', '')


# ### 1.1. Create file structure (if first run-through)
# 
# Execute the following cell **only if you are running this notebook for the first time**. On successive iterations, skip this step.

# In[ ]:


# set a project name
PROJECT_NAME = f'TurbineDetectionYolov7'

# create a new project folder in the Drive with parent directory 'Notebooks'
get_ipython().run_line_magic('mkdir', 'Notebooks/{PROJECT_NAME}')


# In[ ]:


# change the current directory to the project file
get_ipython().run_line_magic('cd', 'gdrive/MyDrive/Notebooks/TurbineDetectionYolov7')


# ### 1.2. Download YOLOv7 repository from GitHub

# In[ ]:


# Clone the YOLOv7 repository to the current directory
get_ipython().system('git clone https://github.com/WongKinYiu/yolov7')

# change the current directory to 'yolov7'
get_ipython().run_line_magic('cd', 'yolov7')

# install the requirements
get_ipython().system('pip install -r requirements.txt')


# #2.0 Import Custom Dataset from API or Formatted .zip Folder
# 
# - *Note:* Use only **one** of the options below for importing custom training data.
# 
# ## *Option 1: Use Roboflow API for Dataset Import*
# ___
# If you have a custom training dataset hosted on Roboflow, you can run the following code snippet below with your API key to import the YOLOv7 formatted dataset.
# 
# 
# 

# In[ ]:


# Install/import Roboflow API
get_ipython().run_line_magic('cd', 'yolov7')
get_ipython().system('pip install roboflow')
from roboflow import Roboflow

# Set to your custom API key
rf = Roboflow(api_key="v1UCLjJfTESikgOTum8T")

# define project from workspace
project = rf.workspace("noahvriese").project("turbinedetection")

# define dataset version and download
dataset = project.version(4).download("yolov7")


# ## *Option 2: Import Formatted .zip Folder*
# ___
# Note that this model requires the following:
# - YOLO .txt annotations
# - Custom .yaml file
# - Correctly formatted directory structure (see below) 
# 
# ├── Notebooks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- The root folder (*must be named 'Notebooks'*)   
# │...├── Custom Turbine Detection.ipynb&nbsp;&nbsp;&nbsp;&nbsp;<- This Google Colab notebook<br>
# │...├── TurbineDetectionYolov7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- The Project folder<br>
# │........└── yolov7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- The YOLOv7 repository folder<br>
# │..............└── **TurbineDetection-4**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- **The YOLOv7 custom dataset folder (.zip)**<br>
# │...................└── train&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Training data split<br>
# │.......................└── labels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │.......................└── images&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │...................└── test&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Test data split<br>
# │.......................└── labels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │.......................└── images&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │...................└── valid&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Validation data split<br>
# │.......................└── labels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │.......................└── images&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │...................└── data.yaml&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- .yaml file for custom model<br>  
# <br>
# 
# ___
# 
# > #### *Note: Custom .YAML File Format*<br> 
# For the example worked in this notebook, the .yaml file for the YOLOv7 architecture consists of 5 lines (see below). 'names' and 'nc' refer to the name, and number of classes of the objects to be detected, 'train' and 'val' refer to the training and validation image folder paths, respectively.
# <br> <br> 
# >  
# > &nbsp;names:<br> 
# > &nbsp;- turbine<br> 
# > &nbsp;nc: 1<br> 
# > &nbsp;train: TurbineDetection-4/train/images<br> 
# > &nbsp;val: TurbineDetection-4/valid/images<br> 
# ___
# 
# <br>
# 
# Once your data and file structure format meet the above criteria, perform the following steps:
# 1. Uploaded your YOLOv7 formatted zip folder into the 'yolov7' directory
# 2. Run the first code cell below to create a custom dataset folder

# In[ ]:


CUSTOM_DATASET_FOLDER = f'YOUR_DATASET_FOLDER_NAME'

get_ipython().run_line_magic('cd', 'yolov7')
get_ipython().run_line_magic('mkdir', '{CUSTOM_DATASET_FOLDER}')


# 3. Next, run the following code cell to extract the .zip folder into your custom dataset folder.

# In[ ]:


import zipfile
filename = os.path.join(os.getcwd(), "{}.zip".format(CUSTOM_DATASET_FOLDER))

# Run to extract zipfile into custom dataset folder
with zipfile.ZipFile(filename, "r") as z_fp:
  z_fp.extractall("./")


# Congratulations! You're all set to begin training the YOLOv7 model on your custom object detection dataset.

# # 3.0 Custom Model Training
# ___
# 
# Once the custom dataset has been imported with the correct formatting via API or .zip folder, we're ready to begin custom model training.
# 
# We will begin by downloading the yolov7_training.pt starting model weights (trained on the COCO dataset). Our custom training job will progressively update these weights as it learns the new image set.

# In[ ]:


# First, ensure we're working within the correct directory: 'yolov7'
get_ipython().run_line_magic('pwd', '')


# In[ ]:


# download starting model weights (COCO dataset)
get_ipython().system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt')


# The code snippet below assumes the dataset was imported via the Roboflow API, if the dataset was imported via zipfile, adjust __'{dataset.location}'__ to the filepath of the __'CUSTOM_DATASET_FOLDER'__ created previously. For example: __*content/gdrive/MyDrive/Notebooks/TurbineDetectionYolov7/yolov7/TurbineDetection-4/data.yaml*__
# <br>
# 
# Also note, the following command assumes a batch size of 16, and will train for 250 epochs. Feel free to adjust these parameters as necessary, but note that excessively large batch sizes tend to produce models which generalize more poorly. A safe general bet is to select a batch size between 16 and 32 images.

# In[ ]:


# run this cell to begin training from initial weights for 250 epochs
get_ipython().system("python train.py --batch 16 --epochs 250 --data {dataset.location}/data.yaml --weights 'yolov7_training.pt' --device 0 ")


# ### 3.1 Google Colab Runtime Timeout
# 
# If training stops due to runtime timeout (runtime is typically limited to 12 hours) follow these steps to resume training from the last saved weights:
# 
# 1. Re-mount google drive
# 2. Navigate to the 'yolov7' folder via command line
# 3. In Google Drive or via CMD, copy **'last.pt'**, from _**/'yolov7/runs/train/weights/last.pt'**_ and move to _**/'yolov7/**_
# 4. Rename **'last.pt'** to **'last_checkpoint.pt'**
# 5. Re-run the training command with updated weight checkpoint (see below):
# 
# **Note:** If runtime timeouts become a significant issue resulting in data loss, excess frustration, etc. see this interesting [StackOverflow discussion](https://stackoverflow.com/questions/54057011/google-colab-session-timeout) regarding tips for overcoming the 12-hour limit.

# In[ ]:


# continue training from last.pt file
get_ipython().system('python train.py --batch 16 --epochs 250 --data {dataset.location}/data.yaml --weights last_checkpoint.pt --device 0 ')


# # 4.0 Evaluate Model Performance
# ___
# Congratulations on finishing your custom object detection model's first training cycle! We can now generate model predictions on the test set and run inference with bounding box confidence displayed for reference.
# 
# ## *4.1 Run Detection Script and Score Confidence*
# We can evaluate the performance of our custom training using the detect.py script within the yolov7 directory.
# 
# **Note:** As usual, we can adjust the arguments in the command below, feel free to adjust the confidence threshold '--conf' as necessary. 
# 
# For extra details on the detect.py script and its arguments, see [this page on the YOLOv7 repository](https://github.com/WongKinYiu/yolov7/blob/main/detect.py#L154).

# In[ ]:


# Run evaluation
get_ipython().system('python detect.py --weights runs/train/exp3/weights/best.pt --conf 0.35 --source {dataset.location}/test/images')


# ## *4.2 Run Inference on Test Set Images*
# 
# We can now visualize the predicted bounding boxes on the test image set as calculated by calling **'detect.py'** above.<br>
# <br>
# This script will loop though the export (**'/exp'**) directory which contains the inferenced images in '.jpg' format. <br>
# <br>
# **Note:** every time **'detect.py'** is executed, a new **'/exp'** folder will be instantiated in the **'/detect'** directory. The trailing digit on the filename will be updated (ex. **'/exp2'**, **'/exp3'**, ...). Adjust the trailing digit as necessary in your filepath.

# In[ ]:


# run inference on images in test set
import glob
from IPython.display import Image, display

# number of images to display
num_images = 50 

i = 0
for imageName in glob.glob('/content/gdrive/MyDrive/Notebooks/TurbineDetectionYolov7/yolov7/runs/detect/exp9/*.jpg'): #assuming JPG
  if i < num_images:
    display(Image(filename=imageName))
    print("\n\n")
  i += 1


# ## *4.3.1 Evaluate Model Perfomance - Yolov7 Performance Plots*
# 
# During training, our yolov7 model framework exports continuous logs of model evaulation metrics. The following code snippet returns the pre-made visualizations of several standard object detection performance metrics.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# get filepath to training data
filepath = f'/content/gdrive/MyDrive/Notebooks/TurbineDetectionYolov7/yolov7/runs/train/exp3/*.png'
num_images = len(glob.glob(filepath))
image_paths = glob.glob(filepath)
image_names = []

for i in range(num_images):
  folders = pd.Series(glob.glob(filepath)[i].split('/'))
  image_names.append(folders.iloc[-1])

i = 0
for i in range(num_images):
  fig = plt.figure(figsize=(10, 10))
  img = mpimg.imread(image_paths[i])
  plt.imshow(img)
  plt.title(image_names[i])
  plt.axis("off")
  plt.show()


# ## *4.3.2 Evaluate Model Performance - Custom Visualizations*
# 
# Alternatively, the Yolov7 model framework produces a **'results.txt'** file which logs performance metrics during model training.
# <br><br>
# **Note:** The default location for this file within the **yolov7/** directory is:
# *'/content/gdrive/MyDrive/Notebooks/[your_project_name]/yolov7/runs/train/exp/__results.txt__'*<br><br>
# 
# We can convert this **.txt** to a **.csv** file for easier import, and use the Pandas and Matplotlib.pyplot to visualize our accuracy and loss metrics.

# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# read in 'results.txt' renamed 'results_epochs_0-249.csv' in this example
results_df = pd.read_csv('results_epochs_0-249.csv', sep=',')


# In[ ]:


# get dataframe shape
results_df.shape


# In[ ]:


# check the dataframe format
results_df.head()


# In[ ]:


# intialize accuracy figure
plt.figure(figsize=(10,8))

# plot precision, recall, and Mean Average Precision at the 0.5 confidence level
plt.plot(results_df.index, results_df['P'])
plt.plot(results_df.index, results_df['R'])
plt.plot(results_df.index, results_df['mAP@.5'])

# label axes and set limits
plt.xlabel('Epochs Trained')
plt.xlim([0,250])
plt.xticks(np.linspace(0,250,11))
plt.ylim([0,1])
plt.yticks(np.linspace(0,1,11))
plt.ylabel('Score (out of 1)')

# label title and produce legend
plt.title('YOLOv7: Accuracy Metrics during Custom Training')
plt.legend(['Precision','Recall', 'mAP: Mean Average Precision'])

# turn grid on for reference
plt.grid()

# show the plot
plt.show()


# In[ ]:


# intialize loss figure
plt.figure(figsize=(10,8))

# plot box/coordinate loss, objectness loss, and total loss
plt.plot(results_df.index, results_df['box'])
plt.plot(results_df.index, results_df['obj'])
plt.plot(results_df.index, results_df['total'])

# label axes and set limits
plt.xlabel('Epochs Trained')
plt.xlim([0,250])
plt.xticks(np.linspace(0,250,11))
plt.ylim([0,0.1])
plt.yticks(np.linspace(0,0.1,11))
plt.ylabel('Loss')

# label title and produce legend
plt.title('YOLOv7: Loss Metrics during Custom Training')
plt.legend(['Box/Coordinate Loss', 'Objectness Loss', 'Total Loss'])

# turn grid on for reference
plt.grid()

# show the plot
plt.show()


# # 5.0 Export Model Weights and Metrics
# ___
# 
# If you would like to deploy your object detection model in another environment, you'll need to export your weights and save them to use later.

# In[ ]:


# zip to download weights and results to a local folder
get_ipython().system('zip -r export.zip runs/detect')
get_ipython().system('zip -r export.zip runs/train/exp3/weights/best.pt')
get_ipython().system('zip export.zip runs/train/exp3/*')


# In[ ]:


get_ipython().system('pip install onnx')
get_ipython().system('pip install onnxsim')
get_ipython().system('pip install netron')


# In[ ]:


get_ipython().system('python export.py --weights runs/train/exp3/weights/best.pt --grid --end2end --simplify         --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640')


# Upon successful execution, the code snippet above will produce the following output:
# <br>
# <br>
# "*Export complete (##.##s). Visualize with https://github.com/lutzroeder/netron.*"
# <br>
# <br>
# Feel free to explore the archetecture of your model in an easy-to-navigate visual format via the **Netron web app** in the linked [Github repository](https://github.com/lutzroeder/netron)

# # Conclusion
# ___
# 
# Congratulations on initializing, training, evaluating, and exporting your custom yolov7 object detection model!<br>
# 
# See the readme in this repository for more model details and the motivations behind creating a wind turbine detection model.

# In[ ]:




