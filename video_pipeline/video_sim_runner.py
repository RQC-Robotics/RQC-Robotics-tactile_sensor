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
train_signal_path = path_config['train_s_video_path']
test_signal_path = path_config['test_s_video_path']

pic_path_len = len(os.path.normpath(pic_path)) + 1
ids = set()


def parse_id(item_path):
    id = item_path[pic_path_len:]
    if '/' in id:
        id = id[:id.find('/')]
    return id


total = 0
for path, folders, files in os.walk(pic_path):
    id = parse_id(path)
    if id:
        ids.add(id)
    total += 1
ids = np.array(sorted(list(ids)))
test_number = config['dataset']['test_items']
if test_number is None:
    test_number = int(len(ids) * config['dataset']['test_frac'])

print(f"Test size is {test_number} items")

test_inds = np.random.choice(len(ids), test_number)
test_ids = set(ids[test_inds])

for path, folders, files in tqdm(os.walk(pic_path), total=total):
    id = parse_id(path)
    new_path = jn(test_signal_path if id in test_ids else train_signal_path,
                  path[pic_path_len:])
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for file_name in files:
        if file_name[:-4] == 'prepared':
            try:
                pic = np.load(jn(path, file_name))
                if file_name.endswith('.npz'):
                    pic = pic['arr_0']
                pic = pic.astype(np.float32)
            except Exception as e:
                print("Can't load file " + jn(path, file_name))
                logging.error(traceback.format_exc())
            dataloader = DataLoader(pic, batch_size=config['sim']['batch_size'])
            signals = []
            for batch in dataloader:
                signal = sim.fiber_real_sim(batch.to(device)).cpu().numpy()
                signals.append(signal)
            if file_name.endswith('npz'):
                np.savez_compressed(jn(new_path, file_name),
                                    np.concatenate(signals))
            else:
                np.save(jn(new_path, file_name), np.concatenate(signals))

# %%
