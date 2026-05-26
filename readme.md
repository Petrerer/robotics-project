What worked was adding this line

import torch
state_dict = torch.load('best_steering_model_xy.pth', map_location='cpu')
torch.save(state_dict, 'best_steering_model_xy_cpu.pth', _use_new_zipfile_serialization=False)

