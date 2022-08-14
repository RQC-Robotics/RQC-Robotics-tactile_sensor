# %%

from pathlib import Path
import sys
import os

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# os.chdir('..')

import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jn
import yaml
import torch_sensor_lib as tsl

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from video_module import Stack_dataset, \
    fit_epoch, eval_epoch, predict, eval_dataset
# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

if not os.path.exists(path_config['reports_path']):
    os.makedirs(path_config['reports_path'])
# %%
torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])

# %%

input_path = path_config['train_s_video_path']
test_input_path = path_config['test_s_video_path']
output_path = path_config['p_video_path']

tr = config['video_train']
frames_number, frames_interval = tr["frames_number"], tr["frames_interval"]

test_dataset = Stack_dataset(output_path, test_input_path, frames_number,
                             frames_interval)
train_dataset = Stack_dataset(output_path, input_path, frames_number,
                              frames_interval, frac=0.7)

# %%

if not torch.cuda.is_available():
    print('CUDA is NOT available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%

signal_shape, pressure_shape = (x.shape for x in train_dataset[0])
print('input chain shape: ', signal_shape, '\noutput chain shape: ',
      pressure_shape)

model_name = tr['model_name']
import models_src

model_class = eval(f"models_src.{model_name}")

args = []
if model_name.startswith("Param"):
    layers = tr['layers']
    args.append(layers)
args.append(frames_number)
args.append(frames_interval)
model = model_class(pressure_shape[-2:], signal_shape[-2:], *args)
model = model.to(device)

# print(model)
optim = torch.optim.Adam(model.parameters(), lr=tr['learning_rate'])
loss_fn = torch.nn.MSELoss()

# %%


def iter_train(train_dataset, test_dataset, model, epochs, optimizer,
               criterion):
    for epoch in range(epochs):
        train_loss = fit_epoch(model, train_dataset, criterion, optimizer,
                               tr['batch_size'], device)
        test_loss = eval_epoch(model, test_dataset, criterion,
                               config['test_batch_size'], device)
        # print("loss", f"{train_loss:.3f}")
        # pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)
        # full_train_loss = eval_dataset(model, train_dataset, criterion, config['test_batch_size'], device)
        # full_test_loss = eval_dataset(model, test_dataset, criterion, config['test_batch_size'], device)
        yield (train_loss, test_loss)


# %%
history = []

# %%
epochs = config['video_train']['epochs']

# %%
with tqdm(total=epochs,
          position=0,
          unit='epoch',
          desc="Learning",
          dynamic_ncols=True) as pbar:
    
    for h in iter_train(train_dataset, test_dataset, model, epochs, optim,
                        loss_fn):
        history.append(h)
        train_loss, test_loss = h
        # print(f"Epoch {i+1}/{total_epochs}",
        #       f"train loss: {full_train_loss:.5f}, test_loss: {full_test_loss:.5f}")
        pbar.update()
        pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)

        titles = ["full_train_loss", "full_test_loss"]
        res = np.array([titles] + history)
        for j, title in enumerate(titles):
            np.savetxt(jn(path_config['reports_path'], title + '.csv'),
                        res[:, j],
                        delimiter=',',
                        fmt='%s')

        os.system('dvc plots show -q')
# %%
full_train_loss, full_test_loss = zip(*history)
res = {
    'train': {
        'loss': full_train_loss[np.argmin(full_test_loss)]
    },
    'test': {
        'loss': min(full_test_loss)
    }
}
with open(jn(path_config['reports_path'], "v_summary.json"), "w") as f:
    json.dump(res, f)

# %%
if not os.path.exists(path_config['v_model_path']):
    os.makedirs(path_config['v_model_path'])
torch.save(model, jn(path_config['v_model_path'], model_name + '.pt'))
