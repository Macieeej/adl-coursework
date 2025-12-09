# adl-coursework

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