import os
import numpy as np
from os.path import join as jn
import yaml

from tqdm import tqdm

import traceback
import logging

## Norms pressure data and cuts parts outside of central circle

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

X, Y = config['env']['sen_geometry']['x_len'], config['env']['sen_geometry'][
    'y_len']
x0, y0 = X // 2, Y // 2
r2 = (x0 * x0)


def func(x2, r2):
    if x2 >= r2:
        return 0
    return np.exp(1 / 8 / ((x2/r2) - 1**2))


hat_func = [[
    0 if (x - x0)**2 + (y - y0)**2 > r2 else func((x - x0)**2 + (y - y0)**2, r2)
    for x in range(X)
]
            for y in range(Y)]
hat_func = np.array(hat_func)
# %%
input_path = path_config['isaac_videos_path']
pic_path = path_config['p_video_path']
inp_path_len = len(os.path.normpath(input_path)) + 1

total = 0
for path, folders, files in os.walk(input_path):
    total += 1

for path, folders, files in tqdm(os.walk(input_path), total=total):
    new_path = jn(pic_path, path[inp_path_len:])
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for file_name in files:
        if file_name[:-4] == 'force':
            try:
                pic = np.load(jn(path, file_name))
                if file_name.endswith('.npz'):
                    pic = pic['arr_0']
                pic *= config['dataset']['mult_coeff']    # norming
                pic *= hat_func    # cutting borders
            except Exception as e:
                print("Can't load file " + jn(path, file_name))
                logging.error(traceback.format_exc())

            np.savez_compressed(jn(new_path, 'prepared.npz'), pic)
            # os.remove(jn(path, file_name))
