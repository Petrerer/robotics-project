What worked was adding this line

import torch
state_dict = torch.load('best_steering_model_xy.pth', map_location='cpu')
torch.save(state_dict, 'best_steering_model_xy_cpu.pth', _use_new_zipfile_serialization=False)


The model to add is _cpu.pth it was trained across 50 epochs.
THe environment.yml was used to make a conda env
important files are train_model.ipynb for training and
live_demo is used for steering the jetbot
