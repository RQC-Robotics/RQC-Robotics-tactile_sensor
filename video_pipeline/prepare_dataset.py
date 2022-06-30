import os
import numpy as np
from os.path import join as jn
import yaml
import torch

import traceback
import logging

## Norms pressure data and cuts parts outside of central circle

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

X, Y = config['env']['sen_geometry']['x_len'], config['env']['sen_geometry']['y_len']
x0, y0 = X // 2, Y // 2
r2 = (x0*x0 + y0*y0)

hat_func = [[0 if (x - x0)**2 + (y - y0)**2 > r2 else 1
             for x in range(X)]
            for y in range(Y)]
hat_func = np.array(hat_func)
# %%
pic_path = path_config['p_video_path']

for path, folders, files in os.walk(pic_path):

    for file_name in files:
        if file_name == 'force.npy':
            try:
                pic = np.load(jn(path, file_name))
                pic *= 40           # norming
                pic *= hat_func     # cutting borders
            except Exception as e:
                print("Can't load file " + jn(path, file_name))
                logging.error(traceback.format_exc())

            np.save(jn(path, 'cutted.npy'), pic)
            os.remove(jn(path, file_name))
