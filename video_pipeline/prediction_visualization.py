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
import h5py
import hdf5plugin

from natsort import natsorted

from video_module import Dynamic_video_dataset, \
    predict, visual_chains
# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

if not os.path.exists(path_config['reports_path']):
    os.makedirs(path_config['reports_path'])

# %%

signal_path = path_config['s_video_path']
pres_path = path_config['p_video_path']

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


def visual_dataset(pres_file, signal_file, step, max_items, begin=0, end=None, duration=30):
    prev_id = ''
    total=min(len(signal_file.keys()) // step, max_items)
    for key in tqdm(natsorted(signal_file.keys())[:total*step:step], total=total):
        
        pressure = pres_file[key][begin:end]
        signal = signal_file[key][begin:end]
        
        id = key[:key.rfind('_')]
        if id != prev_id:
            print(f"\n#### id = {id}", file=file)
            prev_id = id
        prediction = predict(v_model,
                             signal,
                             device)     
        pressure = pressure[-prediction.shape[0]:]
        visual_chains([pressure, prediction],
                      jn(save_path, key))
        print(f"<img src={key+'.gif'} width=400>",
              file=file)



file = open(jn(save_path, "view.md"), 'w')
print("# Visualization", file=file)

for data_path in config['visual']:
    with h5py.File(jn(pres_path, data_path + '.hdf5')) as pres_file, \
        h5py.File(jn(signal_path, data_path + '.hdf5')) as signal_file:
        print(f"\n# {data_path}", file=file)
        visual_dataset(pres_file, signal_file, **config['visual'][data_path])
    

file.close()
