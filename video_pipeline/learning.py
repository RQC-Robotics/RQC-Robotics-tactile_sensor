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

from video_module import Video_dataset, \
    fit_epoch, eval_epoch, predict
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

input_path = path_config['s_video_path']
test_input_path = path_config['test_s_video_path']
output_path = path_config['p_video_path']

test_dataset = Video_dataset(output_path, test_input_path)
train_dataset = Video_dataset(output_path, input_path)

# %%

if not torch.cuda.is_available():
    print('CUDA is NOT available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
tr = config['video_train']
chain_len = tr['chain_len']
test_dataset.split_to_chains(chain_len)
train_dataset.split_to_chains(chain_len)

signal_shape, pressure_shape = (x.shape for x in train_dataset[0])

print('input chain shape: ', signal_shape, '\noutput chain shape: ',
      pressure_shape)

model_name = tr['model_name']
import models_src

model = eval(
    f"models_src.{model_name}(pressure_shape[-2:], signal_shape[-2:]).to(device)"
)
print(model)
optim = torch.optim.Adam(model.parameters(), lr=tr['learning_rate'])
loss_fn = torch.nn.MSELoss()

# %%
print('input chain shape: ', train_dataset[0][0].shape,
      '\noutput chain shape: ', train_dataset[0][1].shape)

# %%


def iter_train(train_dataset, test_dataset, model, epochs, optimizer,
               criterion):
    for epoch in range(epochs):
        train_loss = fit_epoch(model, train_dataset, criterion, optimizer,
                               chain_len, tr['batch_size'], device)
        test_loss = eval_epoch(model, test_dataset, criterion, chain_len,
                               tr['batch_size'], device)
        # print("loss", f"{train_loss:.3f}")
        # pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)
        yield epoch, (train_loss, test_loss)


# %%
history = []

# %%
with tqdm(total=tr['n_epochs'], desc="Learning", unit='epoch',
          ncols=100) as pbar:
    for i, h in iter_train(train_dataset,
                           test_dataset,
                           model=model,
                           epochs=tr['n_epochs'],
                           optimizer=optim,
                           criterion=loss_fn):
        history.append(h)
        print(f"Epoch {i+1}/{tr['n_epochs']}")
        np.savetxt(jn(path_config['reports_path'], 'video_lc.csv'),
                   [['train_loss', 'test_loss']] + history,
                   delimiter=',',
                   fmt='%s')
        os.system('dvc plots show --x-label "epochs" --y-label "loss" -q')
# %%
train_loss, test_loss = zip(*history)
res = {
    'train': {
        'loss': train_loss[np.argmin(test_loss)]
    },
    'test': {
        'loss': min(test_loss)
    }
}
with open(jn(path_config['reports_path'], "v_summary.json"), "w") as f:
    json.dump(res, f)

# %%
if not os.path.exists(path_config['v_model_path']):
    os.makedirs(path_config['v_model_path'])
torch.save(model, jn(path_config['v_model_path'], model_name + '.pt'))
