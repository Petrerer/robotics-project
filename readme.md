# Robotics II

# How to use repo on jetbot
1) Copy file live_demo.ipynb to jetbot
2) Copy file models/dataset_labeled_2/model_to_import.pth to jetbot (lub models/dataset_labeled_1/model_to_import.pth - it was trained on a differently labeled dataset)
3) Fix the path to model from live_demo.ipynb to the uploaded model
4) Run live_demo.ipynb on the track
5) Adjust the parameters from live_demo.ipynb based on how the robot operates

## Dataset
Datasety są w folderze datasets. Można je zmieniać w OUTPUT_DIR w annotation_script.py i w DATASET_NAME w train_model.ipynb

dataset_labeled_1 - wybierałem zawsze +- trzecią kreskę od robota - powinno sprawić że jest bardziej robust
dataset_labeled_2 - wybierałem kreskę jak najdalej żeby jechał trochę bardziej "na krechę"

## How to make the repo work
1) annotation_script.py - Run annotation script on dataset from politechnika because its images are nice but labels are bs
2) train_model.ipynb - Train the model and check the predictions
3) live_demo.ipynb - import the model and run it (models/model_to_import.pth). It's saved in older format which should be compatible with jetbot, its the same model as best.

Notes:
Don't change any packages in the jetbot itself
