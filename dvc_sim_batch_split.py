# %%
import numpy as np
import os
from tqdm import tqdm
import yaml
import math

# %%

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

batch_path = config['sim']['pic_path']
pic_path = config['dataset']['pic_path']
# signal_path = config['dataset']['signal_path']
if not os.path.exists(batch_path):
    os.makedirs(batch_path)

# %%
big_files = os.listdir(pic_path)
n = config['sim']['batch_size']
if n == 'None':
    n = int(config['gengaus']['batch_size'] /
            config["env"]['sen_geometry']['n_angles'] / 2)
i = 0
for file_name in tqdm(big_files, unit='batch'):
    pic = np.load(os.path.join(pic_path, file_name))
    # signal = np.load(jn(signal_path))
    for di in range(math.ceil(pic.shape[0] / n)):
        # inp = signal[i*n:(1+i)*n]
        batch = pic[di * n:(1+di) * n]
        np.save(os.path.join(batch_path, f'batch{i}.npy'), batch)
        i += 1
    # os.remove(os.path.join(pic_path, file_name))
