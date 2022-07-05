# %%

from pathlib import Path
import sys
import os
import dotenv

dotenv.load_dotenv()

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

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import pymongo

ex = Experiment('fiber_sensor_learning_on_videos')
if "username" in os.environ and "host" in os.environ and "password" in os.environ:
    client = pymongo.MongoClient(
        username=os.environ['username'],
        password=os.environ['password'],
        host=os.environ['host'],
        port=27018,
        authSource=os.environ['database'],
        tls=True,
        tlsCAFile=
        "/usr/local/share/ca-certificates/Yandex/YandexInternalRootCA.crt",
    )
    ex.observers.append(
        MongoObserver(client=client, db_name=os.environ['database']))
else:
    ex.observers.append(
        FileStorageObserver('logdir'))

@ex.config
def cfg():
    with open('params.yaml') as conf_file:
        config = yaml.safe_load(conf_file)

# ex.add_config('params.yaml')

@ex.automain
def run(config, _run):
    # %%
    with open('pathes.yaml') as conf_file:
        path_config = yaml.safe_load(conf_file)
    if not os.path.exists(path_config['reports_path']):
        os.makedirs(path_config['reports_path'])

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
    test_dataset.split_to_chains(1)
    train_dataset.split_to_chains(1)

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

    def iter_train(train_dataset, test_dataset, model, epochs, optimizer,
                   criterion, chain_len):
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
    epochs = config['video_train']['epochs']
    total_epochs = 0
    for n_epochs, chain_len in epochs:
        total_epochs += n_epochs

    # %%
    i = 0
    for n_epochs, chain_len in epochs:
        train_dataset.split_to_chains(chain_len)
        test_dataset.split_to_chains(chain_len)
        for _, h in iter_train(train_dataset, test_dataset, model, n_epochs,
                               optim, loss_fn, chain_len):
            history.append(h)
            train_loss, test_loss = h
            print(f"Epoch {i+1}/{total_epochs}",
                  f"train loss: {train_loss:.5f}, test_loss: {test_loss:.5f}")
            _run.log_scalar('train_chain_loss', train_loss)
            _run.log_scalar('test_chain_loss', test_loss)
            _run.log_scalar('chain_len', chain_len)

            np.savetxt(jn(path_config['reports_path'], 'video_lc.csv'),
                       [['train_loss', 'test_loss']] + history,
                       delimiter=',',
                       fmt='%s')
            os.system('dvc plots show --x-label "epochs" --y-label "loss" -q')
            i += 1
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
