# Mask R-CNN for Car Damage Detection and Segmentation [Implemented using the] (https://github.com/matterport/Mask_RCNN)

This is an implementation of Mask R-CNN for Car Damage Detection on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.


The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Source Code of Car Damage Detection in CarDamage.py
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Annonated Dataset build for this project. Dataset was annonated using VIG 1.0.6 on images taken from different sources. (Dataset was   personally collected and is my own property)



# Train the Model on Car Damage Dataset
* [MASK_RCNN_CarDamage_DATASET_Settings.ipynb](https://github.com/M-Tallal-Habib/MASK_RCNN_CarDamage_Detection/blob/master/MASK_RCNN_CarDamage_DATASET_Settings.ipynb) Is the easiest way to start. It shows to train Car Damage model in google colab with all settings done in notebook


* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 


* [Inspect_Car_Damage_Data.ipynb] This notebook visualizes the different pre-processing steps
to prepare the training data.

* [Inspect_Car_Damage_Model.ipynb]This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [Inspect_Car_Damage_Weights.ipynb]
This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns.

### Sample Image 
![](https://github.com/M-Tallal-Habib/MASK_RCNN_CarDamage_Detection/blob/master/CarAccidentDataset/val/4.jpg)

### Mask Detected by Model
![](https://github.com/M-Tallal-Habib/MASK_RCNN_CarDamage_Detection/blob/master/download.png)

### Requirements and Libraries
[All Requirements are already written in requirements.txt](https://github.com/M-Tallal-Habib/MASK_RCNN_CarDamage_Detection/blob/master/requirements.txt)


