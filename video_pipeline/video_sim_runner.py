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
if not os.path.exists(signal_path):
    os.makedirs(signal_path)

# %%
files = os.listdir(pic_path)

# %%
for file_name in files:
    pic = np.load(jn(pic_path, file_name))
    dataloader = DataLoader(pic, batch_size=config['sim']['batch_size'])
    signals = []
    for batch in tqdm(dataloader):
        signal = sim.fiber_real_sim(batch.to(device)).cpu().numpy()
        signals.append(signal)
    np.save(jn(signal_path, file_name), np.concatenate(signals))

# %%
