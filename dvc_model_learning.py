# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jn
import pandas as pd
import yaml
import torch_sensor_lib as tsl

import torch
from torchsummary import summary
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import json

# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

# %%
torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])
seeds = np.random.randint(0, 2**31, size=3)

# %%


class DataSet():

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.file_names = os.listdir(input_path)
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.file_names):
            raise StopIteration
        else:
            res = torch.from_numpy(np.load(jn(self.input_path,
                             self.file_names[self.i]))), \
                  torch.from_numpy(np.load(jn(output_path,
                             self.file_names[self.i])))
            self.i += 1
            return res

    def __getitem__(self, index):
        return np.load(jn(self.input_path, self.file_names[index])), np.load(
            jn(output_path, self.file_names[index]))


input_path = config['dataset']['signal_path']
output_path = config['sim']['pic_path']

test_dataloader = DataSet(jn(input_path, 'test'), output_path)
train_dataloader = DataSet(jn(input_path, 'train'), output_path)

# %% [markdown]
# # defining and fitting torch net

# %%
print('input batch shape: ',
      next(iter(test_dataloader))[0].shape, '\noutput batch shape: ',
      next(iter(test_dataloader))[1].shape)

# %%

if not torch.cuda.is_available():
    print('CUDA is NOT available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
tr = config['train']
model_name = tr['model_name']
model = eval(
    f"tsl.{model_name}(next(iter(train_dataloader))[0].shape[1:], next(iter(train_dataloader))[1].shape[1:]).to(device)"
)
# summary(model, next(iter(train_dataloader))[0].shape[1:], device=device)
# print(model)
optim = torch.optim.Adam(model.parameters(), lr=tr['learning_rate'])
loss_fn = torch.nn.MSELoss()

# %%
# training functions


def fit_epoch(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    processed_data = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    return train_loss


def eval_epoch(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    processed_size = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    return val_loss


def predict(model, test_loader):
    with torch.no_grad():
        result = []
        # test_loader = DataLoader(test_dataset,
        #                          batch_size=batch_size,
        #                          shuffle=False)
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            result.append(outputs)

    return torch.cat(result).numpy()


def iter_train(train_loader, test_loader, model, epochs, optimizer, criterion):
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with tqdm(total=epochs, desc="Learning", unit='epochs') as pbar:
        for epoch in range(epochs):
            train_loss = fit_epoch(model, train_loader, criterion, optimizer)
            test_loss = eval_epoch(model, test_loader, criterion)
            # print("loss", f"{train_loss:.3f}")
            pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)
            pbar.update(1)
            yield epoch, (train_loss, test_loss)


# %%
history = []

# %%
for i, h in iter_train(train_dataloader,
                       test_dataloader,
                       model=model,
                       epochs=tr['n_epochs'],
                       optimizer=optim,
                       criterion=loss_fn):
    history.append(h)
    # train_loss, test_loss = zip(*history)
    # if i%10 == 0:
    #     # clear_output(wait=True)
    #     plt.figure(figsize=(12, 8))
    #     plt.plot(train_loss, label="train_loss")
    #     plt.plot(test_loss, label="test_loss")
    #     plt.legend(loc='best')
    #     plt.xlabel("epochs")
    #     plt.ylabel("loss")
    #     plt.show()
# %%
train_loss, test_loss = zip(*history)
df = pd.DataFrame({"train_loss": train_loss, 'test_loss': test_loss})
if not os.path.exists(config['evaluate']['reports_path']):
    os.makedirs(config['evaluate']['reports_path'])
df.to_csv(jn(config['evaluate']['reports_path'], 'learning_curve.csv'),
          index=False)
res = {'train': {'loss': train_loss[-1]}, 'test': {'loss': test_loss[-1]}}
with open(jn(config['evaluate']['reports_path'], "summary.json"), "w") as f:
    json.dump(res, f)

# %%
# train_loss, test_loss = zip(*history)
# plt.figure(figsize=(12, 8))
# plt.plot(train_loss, label="train_loss")
# plt.plot(test_loss, label="test_loss")
# plt.legend(loc='best')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()

# %%
os.makedirs(tr['models_path'])
torch.save(model, jn(tr['models_path'], model_name + '.pt'))

# %% [markdown]
# # evaluate model on don't seen data

    # # %%
    # model = torch.load(jn(tr['models_path'], model_name + '.pt'))
    # model.eval()
    # # %%
    # predictions = predict(model, test_dataloader)
    # predictions.shape

    # # %%
    # print(
    #     f"Test loss is {loss_fn(torch.from_numpy(predictions), torch.from_numpy(np.concatenate([x for _, x in test_dataloader]))).item():.4f}"
    # )

    # # %%
    # print('len= ', len(predictions))
    # N = 8    # number of example
    # plt.imshow(predictions[N])
    # plt.show()
    # plt.imshow([x for _, x in test_dataloader][0][N])
    # plt.show()
