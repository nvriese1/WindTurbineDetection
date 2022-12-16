#!/usr/bin/env python
# coding: utf-8

# # Capstone Project 3: Custom Wind Turbine Detection using YOLOv7 Architecture
# 
# ## *Notebook 1/3: Data Wrangling & Exploratory Data Analysis*
# 
# This notebook details the initial data wrangling required to re-format the image and label data from a Kaggle dataset of interest for compatilibity with the YOLOv7 (You Only Look Once) object detection framework (built in Pytorch). The starting aerial imagery dataset can be acquired from [Kaggle - Wind Turbine Detection](https://www.kaggle.com/datasets/saurabhshahane/wind-turbine-obj-detection). 
# 
# Upon download, the zipfile containing subfolders '**images**' and '**labels**' should be placed in a directory '**data/raw/**' for optimal code execution.
# 
# ### *Steps Covered in this Notebook*
# 
# **0.0** &nbsp;&nbsp;Import required modules & libraries
# 
# **1.0** &nbsp;&nbsp;Create data directory structure
# 
# **2.0** &nbsp;&nbsp;Extract raw dataset from zipfile
# 
# **3.0** &nbsp;&nbsp;Define helper functions & wrangle image data
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**3.1**&nbsp;&nbsp;Define helper functions
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**3.2**&nbsp;&nbsp;Wrangle image data
#     
# **4.0** &nbsp;&nbsp;Perform train-test split
# 
# **5.0** &nbsp;&nbsp;Copy train-test-validate data to subfolders
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**5.1**&nbsp;&nbsp;Copy label data to train-test-valid split subfolders
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**5.2**&nbsp;&nbsp;Copy image data to train-test-valid split subfolders
# 
# **6.0** &nbsp;&nbsp;Create train-test-validate Pytorch datasets
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**6.1**&nbsp;&nbsp;Create annotation .csv for train-test-valid splits
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**6.2**&nbsp;&nbsp;Define custom Pytorch dataset class
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**6.3**&nbsp;&nbsp;Image augmentation: define data transform sequence
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**6.4**&nbsp;&nbsp;Create Pytorch train-test-validation datasets
# 
# **7.0** &nbsp;&nbsp;Load image datasets & visualize
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**7.1**&nbsp;&nbsp; Load image batch from train/test/validation splits
#     <br>
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**7.2**&nbsp;&nbsp; Visualize loaded image data
# ___

# ## 0.0 Import Required Modules & Libraries

# In[387]:


# import required modules and libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import shutil

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split


# ___
# ## 1.0 Create Data Directory Structure
# 
# As stated in the introduction above, the zipped [Kaggle - Wind Turbine Detection](https://www.kaggle.com/datasets/saurabhshahane/wind-turbine-obj-detection) dataset, or other custom training dataset of interest, should be downloaded into the following folder structure: '**data/raw/**'

# In[ ]:


get_ipython().run_line_magic('mkdir', 'data')
get_ipython().run_line_magic('cd', 'data')
get_ipython().run_line_magic('mkdir', 'raw')
get_ipython().run_line_magic('cd', 'raw')


# In[385]:


# print the current working directory
print(os.getcwd())


# Once we've checked that our root files are in place, we will create **test**, **train**,and **valid** folders for our test, train, and validation data splits, respectively.

# In[221]:


# create test, train, and validation folders
get_ipython().run_line_magic('mkdir', 'test')
get_ipython().run_line_magic('mkdir', 'train')
get_ipython().run_line_magic('mkdir', 'valid')


# In[223]:


# list the current contents of the 'raw' folder to check new directories
get_ipython().run_line_magic('ls', '')


# Now we can create subdirectories **images** and **labels** within each of our test, train, and validation directories. These will store the corresponding image and label sets for each data split.

# In[226]:


# change to test directory and create image, labels folders
get_ipython().run_line_magic('cd', 'test')
get_ipython().run_line_magic('mkdir', 'images')
get_ipython().run_line_magic('mkdir', 'labels')
get_ipython().run_line_magic('cd', '..')

# change to train directory and create image, labels folders
get_ipython().run_line_magic('cd', 'train')
get_ipython().run_line_magic('mkdir', 'images')
get_ipython().run_line_magic('mkdir', 'labels')
get_ipython().run_line_magic('cd', '..')

# change to valid directory and create image, labels folders
get_ipython().run_line_magic('cd', 'valid')
get_ipython().run_line_magic('mkdir', 'images')
get_ipython().run_line_magic('mkdir', 'labels')
get_ipython().run_line_magic('cd', '..')


# ___
# ## 2.0 Extract Raw Dataset from Zipfile
# 
# The following two code cells allow for dataset extraction. If necessary, adjust the **CUSTOM_DATASET_FOLDER** string to match the **name of the custom dataset** you wish to extract into the  '**data/raw/**' file.

# In[5]:


CUSTOM_DATASET_FOLDER = f'kaggle_wind_turbines'


# In[12]:


# get full filepath of custom dataset folder
filename = os.path.join(os.getcwd(), "{}.zip".format(CUSTOM_DATASET_FOLDER))

# run to extract zipfile into custom dataset folder
with zipfile.ZipFile(filename, "r") as z_fp:
    z_fp.extractall("./")


# Once the dataset folder has been unzipped, let's check for it in the current directory.

# In[227]:


get_ipython().run_line_magic('ls', '')


# ___
# ## 3.0 Define Helper Functions & Wrangle Image Data
# 
# ### 3.1 Define Helper Functions
# 
# The following functions optimize the file wrangling process for loading image data into a Pytorch dataset class. See the docstrings associated with each function for a description of the operation performed.

# In[1]:


def get_null_labels(image_dir='images', label_dir='labels'):
    """
    Finds images with no corresponding .txt label (null images) and returns a list
    of null image filenames (list of strings). One string is returned for each null image.
    
    Arguments: (2)
    1. image_dir: name of directory containing image files (.jpg, .bmp, etc.)
    2. label_dir: name of directory containing label files in YOLO format (.txt)
    
    Requirements: (2)
    1. Image filenames in 'image_dir' must match the YOLO .txt labels in 'label_dir'
    2. Directories 'image_dir' and 'label_dir' must be in the current working directory
    """
    # get list of image filenames in the image directory
    image_names = [i for i in os.listdir(image_dir)]
    # get image filenames without extension (ex: .jpg, .bmp)
    images_no_ext = [i.split('.')[0] for i in image_names]
    
    # get list of label filenames in the label directory
    label_names = [i for i in os.listdir(label_dir)]
    # get label filenames without extention (ex: .txt)
    labels_no_ext = [i.split('.')[0] for i in label_names]

    # create empty list
    null_labels = []

    # iterate over image filenames
    for name in images_no_ext:
        # if image filename in labels, continue
        if name in labels_no_ext:
            continue
        # else, append the name of the image filename to null_labels (image is null)
        else:
            null_labels.append(name)
            
    return null_labels


# In[71]:


def write_null_labels(null_label_list):
    """
    Creates an empty .txt file (YOLO label format) for all null labels in 'null_label_list'.
    
    Arguments: (1)
    1. null_label_list: list of strings containing null filenames (returned by get_null_labels)
    
    Requirements: (1)
    1. Directory 'labels' must be in the current working directory
    """
    # iterate over null labels (null images)
    for name in null_labels:
        # create an empty .txt label file with the corresponding image filename
        f = open(os.path.join('labels','{}.txt'.format(name)), 'x')
        # close the file
        f.close()
        
    return


# In[102]:


def clean_extra_labels():
    """
    Deletes extraneous labels in folder 'labels/' if no image exists for the label name.
    
    Arguments: (0)
    
    Requirements: (1)
    1. Directories 'image_dir' and 'label_dir' must be in the current working directory
    """
    # get list of labels with no corresponding images present in the images directory
    null_list = get_null_labels(image_dir='labels', label_dir='images')
    
    # remove the extraneous labels from the label folder
    for f in null_list:
        os.remove(os.path.join('labels','{}.txt'.format(f)))
        
    return


# In[234]:


def get_images_labels(sort=True):
    """
    Returns two lists of strings: 'images', and 'labels' containing 
    filenames for images and labels in the dataset.
    
    Arguments: (1)
    1. sort: if True, returns a sorted list of images and labels
    
    Requirements: (1)
    1. Directories 'images' and 'labels' must be in the current working directory
    """
    # get list of image filenames in 'images/'
    images = [os.path.join('images', x) for x in os.listdir('images')]
    # get list of label filenames in 'labels/'
    labels = [os.path.join('labels', x) for x in os.listdir('labels') if x[-3:] == "txt"]
    
    # sort, if required
    if sort==True:
        images.sort()
        labels.sort()
    
    return images, labels


# In[1]:


def make_annotations_from_labels(labels):
    """
    Takes a list of strings referencing YOLO format .txt labels, and
    returns a DataFrame with the following columns:
    'filename', 'class', 'x', 'y', 'height', 'width'.
    
    Arguments: (1)
    1. labels: list of strings of filenames for YOLO format .txt labels
    
    Requirements: (0)
    """
    # create empty list
    annot_list = []
    
    # iterate over filenames in labels list
    for name in labels:
        
        # open the label .txt file in read configuration
        f = open(os.path.join('{}'.format(name)), 'r')
        
        # check if the .txt file is empty (is a null label)
        check_file = os.stat('{}'.format(name)).st_size
        
        # if the file isn't a null label
        if(check_file != 0):
            
            # iterate over lines in the file
            num_lines = 0
            for line in f:
                
                # split each line and extract features
                num_lines += 1
                data = line.split()
                temp = name.split('.')[0]
                filename = '{}.jpg'.format(temp.split('\\')[-1])
                
                # assign class, x, y, height, and width to variables
                cls = int(data[0])
                x = float(data[1])
                y = float(data[2])
                height = float(data[3])
                width = float(data[4])
                
                # append label data components to the annotation list 
                annot_list.append([filename, cls, x, y, height, width])
                
        # close the file
        f.close()
    
    # create a DataFrame containing all label annotation data
    annotations = pd.DataFrame(annot_list, columns=['filename','class','x', 'y', 'height', 'width'])
    return annotations


# In[276]:


def copy_to_subdirectories(src, dst, list_of_files):
    """
    Copies a list of files references by list of string filenames from a source directory 
    to a destination directory
    
    Arguments (3):
    1. src: name of source directory/folder
    2. dst: name of destination directory/folder
    
    Requirements (1):
    1. 'list_of_files' is a string list of filenames present in directory 'src'
    """

    source_folder = src
    destination_folder = dst
    
    # get filenames from list_of_files
    files_to_move = [list_of_files[i].split('\\')[-1] for i in range(0,len(list_of_files))]

    # iterate over files
    for file in files_to_move:
        
        # construct full file path
        source = source_folder + file
        destination = destination_folder + file
        
        # move file
        shutil.copy2(source, destination)
        print('Moved:', file)


# ### 3.2 Wrangle Image Data
# 
# The following operations utilize the helper functions defined above to pre-process and wrangle image data into a usable format for train-test splitting and the Pytorch dataset class.
# 
# The operations begin by ensuring the current working directory is '**data/raw/**' to meet function requirements.

# In[111]:


get_ipython().run_line_magic('pwd', '')


# In[100]:


# get the list of null images (no objects of interest present, ex. 0 wind turbines)
null_list = get_null_labels('images', 'labels')


# In[103]:


# write null .txt labels for each null image in YOLO annotation format
write_null_labels(null_list)


# In[ ]:


# remove .txt labels that reference images not present in the 'images/' directory
clean_extra_labels()


# In[235]:


# get lists of image and label filenames
images, labels = get_images_labels()


# ___
# ## 4.0 Perform Train-Test Split
# 
# The following operation performs a (80/10/10) train-test-validate split on the image and label dataset. 
# 
# The list of images and labels returned by get_images_labels() is used as input for scikit-learn's train_test_split. The split operation is repeated on the test set to produce a validation split.

# In[237]:


# split the dataset (80/20) into train-test splits 
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size = 0.2, random_state = 1)

# split the test dataset again (50/50) into test-validate splits
val_images, test_images, val_labels, test_labels = train_test_split(val_images, val_labels, test_size = 0.5, random_state = 1)


# In[257]:


# check one of the resulting outputs: test_labels
print(test_labels)


# ___
# ## 5.0 Copy Train-Test-Validate Data to Subfolders (YOLOv7 File Structure)
# 
# In order to train the YOLOv7 object detection model on a new custom image dataset, the following file structure is required for train-test-validation image data and label annotations:
# 
# │.........└── data<br>
# │..............└── raw&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Current working directory<br>
# │...................└── _images_&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Raw image data (to be copied to train/test/valid subdirectories)<br>
# │...................└── _labels_&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Raw label data (to be copied to train/test/valid subdirectories)<br>
# │...................└── **train**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Training data split subfolder<br>
# │.......................└── labels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │.......................└── images&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │...................└── **test**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Test data split subfolder<br>
# │.......................└── labels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │.......................└── images&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │...................└── **valid**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<- Validation data split subfolder<br>
# │.......................└── labels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# │.......................└── images&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# 
# The following cells will use the file lists returned by the train-test-validation split to copy data (images and label annotations) from the '*image/*' and '*labels/*' folders in the current working directory '*raw/*', to the relevant subdirectories (**train/**, **test/**, **valid/**) in accordance with the file assignment.

# ### 5.1 Copy Label Data to Train-Test-Valid Split Subfolders

# In[ ]:


# set filepath for label data source (label directory with cleaned image labels)
src = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/labels/'


# In[ ]:


# copy files from test_labels to appropriate subdirectories in accordance with YOLOv7 file structure

# define destination file path
dst = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/test/labels/'

# copy labels in the test split to the appropriate subdirectories
copy_to_subdirectories(src, dst, test_labels)


# In[278]:


# copy files from train_labels to appropriate subdirectories in accordance with YOLOv7 file structure

# define destination file path
dst = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/train/labels/'

# copy labels in the train split to the appropriate subdirectories
copy_to_subdirectories(src, dst, train_labels)


# In[279]:


# copy files from val_labels to appropriate subdirectories in accordance with YOLOv7 file structure

# define destination file path
dst = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/valid/labels/'

# copy labels in the validation split to the appropriate subdirectories
copy_to_subdirectories(src, dst, val_labels)


# ### 5.2 Copy Image Data to Train-Test-Valid Split Subfolders

# In[ ]:


# set filepath for image data source (raw image directory)
src = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/images/'


# In[281]:


# copy test_images to appropriate subdirectories for YOLOv7 structure

# define destination file path
dst = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/test/images/'

# copy images in the test split to the appropriate subdirectories
copy_to_subdirectories(src, dst, test_images)


# In[280]:


# copy train_images to appropriate subdirectories for YOLOv7 structure

# define destination file path
dst = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/train/images/'

# copy images in the train split to the appropriate subdirectories
copy_to_subdirectories(src, dst, train_images)


# In[282]:


# copy val_images to appropriate subdirectories for YOLOv7 structure

# define destination file path
dst = r'C:/Users/Charles/Desktop/Springboard/Projects/Capstone_Three/data/raw/valid/images/'

# copy images in the validation split to the appropriate subdirectories
copy_to_subdirectories(src, dst, val_images)


# ___
# ## 6.0 Create train-test-validate Pytorch datasets
# 
# The following section executes the steps required to load data from the train-test-validation images/labels/ file structure into a Pytorch dataset class.  
# 
# ### 6.1 Create Annotation .CSVs for Train-Test-Valid Splits
# 
# The Pytorch dataset class requires an annotation .CSV file describing the filenames and location/size of bounding box annotations within the custom dataset. The following section creates the annotation .CSV for each of the train-test-validation data splits. 

# In[302]:


get_ipython().run_line_magic('cd', 'test')

# get annotations DataFrame from .txt files
test_annotations = make_annotations_from_labels(test_labels)

# create annotation .csv from DataFrame for Pytorch dataset compatibility
test_annotations.to_csv('test_annotations.csv', index=False, encoding='utf-8')

get_ipython().run_line_magic('cd', '..')


# In[303]:


get_ipython().run_line_magic('cd', 'train')

# get annotations DataFrame from .txt files
train_annotations = make_annotations_from_labels(train_labels)

# create annotation .csv from DataFrame for Pytorch dataset compatibility
train_annotations.to_csv('train_annotations.csv', index=False, encoding='utf-8')

get_ipython().run_line_magic('cd', '..')


# In[304]:


get_ipython().run_line_magic('cd', 'valid')

# get annotations DataFrame from .txt files
valid_annotations = make_annotations_from_labels(val_labels)

# create annotation .csv from DataFrame for Pytorch dataset compatibility
valid_annotations.to_csv('valid_annotations.csv', index=False, encoding='utf-8')

get_ipython().run_line_magic('cd', '..')


# ### 6.2 Define Custom Pytorch Dataset Class
# 
# The following code defines the 'TurbineDataset' Pytorch dataset class and relevant methods 'len' and 'getitem'.

# In[305]:


class TurbineDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# ### 6.3 Image Augmentation: Define Data Transform Sequence
# 
# The following code establishes a rudimentary image augementation pipeline 'transform_seq', which will be implemented on the loaded train/test/validate data splits (images and labels) upon dataset creation (6.4).

# In[367]:


transform_seq = torch.nn.Sequential(
    
    # resize image to 640x640 for YOLOv7 model architecture
    transforms.Resize(640),
    # randomized rotation +/- 90 deg
    transforms.RandomRotation(90),
    # randomized horizontal flip
    transforms.RandomHorizontalFlip([0.5])
    # adjust brightness +/- 25%
    transforms.adjust_brightness([0.25])
    
)


# ### 6.4 Create Pytorch Train-Test-Validation Datasets
# 
# We can now finally create our test/train/validation datasets using the defined Pytorch TurbineDataset class. Image and label transformations as defined in 6.3 are applied automatically during dataset creation.

# In[368]:


test_dataset  = TurbineDataset('test/test_annotations.csv', 'test/images', transform=transforms_seq, target_transform=transforms_seq)
train_dataset = TurbineDataset('train/train_annotations.csv', 'train/images', transform=transform_seq, target_transform=transforms_seq)
valid_dataset = TurbineDataset('valid/valid_annotations.csv', 'valid/images', transform=transforms_seq, target_transform=transforms_seq)


# In[369]:


# check the number of images in the test dataset 
test_dataset.__len__()


# In[370]:


# check the number of images in the train dataset 
train_dataset.__len__()


# In[371]:


# check the number of images in the validation dataset 
valid_dataset.__len__()


# In[378]:


image,label = train_dataset.__getitem__(0)


# In[382]:


print(image, label)


# ___
# ## 7.0 Load Image Datasets & Visualize
# 
# This section details several steps for loading a created Pytorch dataset and visualizing the image/label data contained within. 
# 
# ### 7.1 Load Image Batch from Train/Test/Validation Splits
# 
# This cell defines an image batch size of 16 images/batch and creates Pytorch dataloaders for each of the train/test/validation datasets.

# In[372]:


BATCH_SIZE = 16

test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ### 7.2 Visualize Loaded Image Data
# 
# To ensure our data was correctly loaded and transformed, we will visualize a single image and label from the train_dataloader.

# In[373]:


# Display a single train image and label.
train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# remove single-dimensional entry from train_features, assign to img
img = train_features[0].squeeze()
# get label from dataloader
label = train_labels[0]

# show the loaded image and label
plt.imshow(img.T)
plt.show()
print(f"Label: {label}")


# ___
# ## _Conclusion_
# 
# Congratulations on succesfully cleaning, wrangling, and loading your raw custom image dataset into a defined Pytorch dataset class. <br>
# 
# In the next notebook (executed in Google Colab), we will train the YOLOv7 object detection model to identify the wind turbine class in aerial imagery using the cleaned dataset produced in this exercise.

# In[ ]:




