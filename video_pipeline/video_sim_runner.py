# %%
import os

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from tqdm import tqdm
import torch_sensor_lib as tsl
from torch.utils.data import DataLoader

import numpy as np
from os.path import join as jn
import yaml
import torch

import traceback
import logging
# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

seed = np.random.seed(config['random_seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
sim = tsl.FiberSimulator(config, device=device)

# %%
pic_path = path_config['p_video_path']
signal_path = path_config['s_video_path']
pic_path_len = len(os.path.normpath(pic_path)) + 1
total = 0
for path, folders, files in os.walk(pic_path):
    total += 1

for path, folders, files in tqdm(os.walk(pic_path), total=total):
    new_path = jn(signal_path, path[pic_path_len:])
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for file_name in files:
        if file_name == 'prepared.npy':
            try:
                pic = np.load(jn(path, file_name)).astype(np.float32)
            except Exception as e:
                print("Can't load file "+jn(path, file_name))
                logging.error(traceback.format_exc())
            dataloader = DataLoader(pic, batch_size=config['sim']['batch_size'])
            signals = []
            for batch in dataloader:
                signal = sim.fiber_real_sim(batch.to(device)).cpu().numpy()
                signals.append(signal)
            np.save(jn(new_path, file_name), np.concatenate(signals))

# %%
