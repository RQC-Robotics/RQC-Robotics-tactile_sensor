
from pathlib import Path
import sys
import os

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jn
import yaml

import torch
from tqdm import tqdm
import json

from video_module import Video_dataset, \
    predict, visual_chains
# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

if not os.path.exists(path_config['reports_path']):
    os.makedirs(path_config['reports_path'])

# %%

test_input_path = path_config['test_s_video_path']
output_path = path_config['p_video_path']

test_dataset = Video_dataset(output_path, test_input_path)

# %%

if not torch.cuda.is_available():
    print('CUDA is NOT available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
tr = config['video_train']

# test_dataset.split_to_chains(300)

v_model_path = path_config['v_model_path']
v_model = torch.load(v_model_path+'/'+config['video_train']['model_name']+'.pt', 
    map_location=device)

save_path = path_config['video_predict_vis_path']

for pressure, signal, file_name in zip(test_dataset.pressure, 
    test_dataset.signal, test_dataset.files):
    file_name = file_name[:-4]
    begin = 0
    prediction = 50*predict(v_model, signal[begin:], device, initial_pressure=pressure[0+begin])
    pressure = pressure[-prediction.shape[0]:]
    visual_chains([pressure, prediction], jn(save_path, file_name.replace('/', '_')))

