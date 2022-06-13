# %%
from tqdm import tqdm
import torch_sensor_lib as tsl

import numpy as np
from os.path import join as jn
import yaml
import torch
import os
'''
Requirements:

dataset.signal_path
sim.pic_path
random_seed

env.sen_geometry
env.phys


dataset.signal_path, dataset.pic_path, random_seed, env.sen_geometry, env.phys
'''
# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
seed = np.random.seed(config['random_seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
sim = tsl.FiberSimulator(config, device=device)

# %%
pic_path = config['sim']['pic_path']
signal_path = config['dataset']['signal_path']
if not os.path.exists(signal_path):
    os.makedirs(jn(signal_path, 'test'))
    os.makedirs(jn(signal_path, 'train'))

# %%
test_size = config["sim"]["test_size"]
if test_size == 'None':
    test_size = len(os.listdir(pic_path)) // 10
# %%
for i, file_name in enumerate(tqdm(os.listdir(pic_path))):
    pic = np.load(jn(pic_path, file_name))
    signal = sim.fiber_real_sim(pic).cpu().numpy()
    if i < test_size:
        np.save(jn(signal_path, 'test', file_name), signal)
    else:
        np.save(jn(signal_path, 'train', file_name), signal)

# %%
