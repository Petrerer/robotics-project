# Robotics II

# How to use repo on jetbot

1) przekopiować pliki live_demo.ipynb i jeden z modeli z models/dataset_name/model_to_import
(ważne - model to import jest zapisany w starym formacie pasujacym do jetbota)
2) odpalić live_demo.ipynb i ztestowac jetbota
3) za pomocą slajderów w notebooku przesuwać parametry jetbota

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