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

input_path = path_config['train_s_video_path']
test_input_path = path_config['test_s_video_path']
output_path = path_config['p_video_path']

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
v_model = torch.load(v_model_path + '/' + config['video_train']['model_name'] +
                     '.pt',
                     map_location=device)

save_path = path_config['video_predict_vis_path']
if not os.path.exists(save_path):
    os.makedirs(save_path)


def visual_dataset(dataset, step, max_items, begin=0):
    i = 0
    prev_id = ''
    for pressure, signal, file_name in tqdm(
            zip(dataset.pressure[::step], dataset.signal[::step],
                dataset.files[::step]),
            total=min(len(dataset.files) // step, max_items)):
        i += 1
        file_name = file_name[:-4]
        id = file_name[:file_name.rfind('/')]
        id = id[:id.rfind('/')]
        if id != prev_id:
            print(f"\n#### id = {id}", file=file)
            prev_id = id
        prediction = predict(v_model,
                             signal[begin:],
                             device,
                             initial_pressure=pressure[0 + begin])
        pressure = pressure[-prediction.shape[0]:]
        visual_chains([pressure, prediction],
                      jn(save_path, file_name.replace('/', '_')))
        print(f"<img src={file_name.replace('/', '_')+'.gif'} width=400>",
              file=file)
        if i == max_items:
            break


file = open(jn(save_path, "view.md"), 'w')
print("# Visualization", file=file)

test_dataset = Video_dataset(output_path, test_input_path)
print("# Test dataset", file=file)
visual_dataset(test_dataset, **config['visual']['test'])

if 'train' in config['visual'] and config['visual']['train']['max_items'] > 0:
    train_dataset = Video_dataset(output_path, input_path)
    print("\n\n # Train dataset", file=file)
    visual_dataset(train_dataset, **config['visual']['train'])
file.close()
