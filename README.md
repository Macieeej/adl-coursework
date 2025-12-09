# Human Visual Saliency Prediction with Multi-Resolution CNN

### Applied Deep Learning Coursework | University of Bristol

This project re-implements and extends a state-of-the-art Multi-Resolution Convolutional Neural Network (Mr-CNN) for predicting human eye fixation (visual saliency) in natural images. The model learns to identify where humans are most likely to look when viewing a scene, a core problem in computer vision, human–computer interaction, and attention modeling.

The implementation closely follows the architecture introduced in Predicting Eye Fixations using Convolutional Neural Networks, with large-scale hyperparameter tuning, checkpointing, evaluation, and transfer learning.

⸻

🧠 Problem Overview

Human visual attention focuses on only a small subset of the information present in a scene. Accurately predicting saliency regions (eye fixation points) has applications in:
	•	Image and video compression
	•	Autonomous driving
	•	Medical imaging
	•	Human–computer interaction
	•	Robotics and surveillance

Traditional saliency methods rely on hand-crafted features and fail to generalize across complex scenes. This project addresses these limitations using a deep, multi-resolution CNN architecture that learns hierarchical saliency features directly from data.

⸻

🏗️ Model Architecture

The implemented Mr-CNN consists of:
	•	Three parallel input streams, each processing the image at a different resolution
	•	Each stream contains:
  	•	3× Convolution + Pooling blocks
  	•	Dropout regularization
  	•	Fully connected layers
  •	The streams are concatenated and passed through:
  	•	A final fully connected layer
  	•	A binary classifier predicting fixation vs non-fixation

Input Representation
	•	Each training sample is a 42×42 RGB image patch
	•	Binary classification:
	•	Fixation patch: saliency > 0.9
	•	Non-fixation patch: saliency < 0.1

This multi-resolution design enables the model to capture both local fine details and global scene context.



### 🏃 Run Instructions

Make sure to edit the dir global variables at the top of the program to point to the directory that contains the data and where the checkpoints will be saved/loaded from, the name may be different. On BC4, you will probably have this in the work directory.

Extensions:
Download Nusef and Toronto datasets from: http://saliency.mit.edu/datasets.html
Create a parent folder, with the folowing folders: adl-coursework with all the python scripts, MIT_data, ALLFIXATIONMAPS with ground truth fixation maps for MIT, TORONTO_database with all images in toronto dataset, ALLFIXATIONMAPS_TORONTO with all fixation maps for toronto, NUSEF_database with all the files from the folders downloded from http://saliency.mit.edu/datasets.html for NUSEF, ALLFIXATIONMAPS_NUSEF an empty folder into which ground truth saliency maps will be added through a script, CHECKPOINTS dir into which model's checkpoints will be save for transfer learning

Prepare data:

python nusef_to_mit.py
put the resulting ALLFIXATIONMAPS_NUSEF file into the parent directory
python create_test_val_dataset.py --test-or-val 'toronto'
python create_train_test_val_split.py
python create_train_dataset.py
python create_test_val_dataset.py --test-or-val 'test'
python create_test_val_dataset.py --test-or-val 'val'


Zero-shot:
python test_on_toronto.py --checkpoint-name "<the name of the checkpoint which performs best, stored in CHECKPOINTS dir>"
python test_on_nusef.py --checkpoint-name "<the name of the checkpoint which performs best, stored in CHECKPOINTS dir>"

Transfer Learning:
python mr_cnn_transfer.py --checkpoint-name "<the name of the checkpoint which performs best, stored in CHECKPOINTS dir>"

Running on BC4:

Make sure all your dataset directories are set and use the train.sh script to run training of the model with default parameters.
Use the testCheckpoint.sh script to test the given checkpoint and parameters.
NOTE: The code works but the script may have difficulties finding the file especially if you used a symlink despite the fact it is factually correct, this is 50% a user error and 50% a python problem.

The above extension commands can be run as well on BC4 by editing the train.sh script to run them instead of mr_cnn.py
