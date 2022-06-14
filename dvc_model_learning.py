# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jn
import pandas as pd
import yaml
import torch_sensor_lib as tsl

import torch
from torchinfo import summary
from tqdm import tqdm
import os
import json

# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

# %%
torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])
seeds = np.random.randint(0, 2**31, size=3)

# %%


class DataLoader():

    def __init__(self, input_path, output_path, batch_size=None):
        self.input_path = input_path
        self.output_path = output_path
        self.file_names = os.listdir(input_path)
        if batch_size is None:
            self.batch_size = np.load(jn(self.input_path,
                                         self.file_names[0])).shape[0]
        else:
            self.batch_size = batch_size
        self.i = 0
        self.loaded_in = []
        self.loaded_out = []
        self.loaded_len = 0

    def __iter__(self):
        self.i = 0
        self.loaded_in = []
        self.loaded_out = []
        self.loaded_len = 0
        return self

    def __next__(self):
        while (self.i < len(self.file_names)
               and self.loaded_len < self.batch_size):
            self.loaded_in.append(
                np.load(jn(self.input_path, self.file_names[self.i])))
            self.loaded_out.append(
                np.load(jn(output_path, self.file_names[self.i])))
            self.loaded_len += self.loaded_in[-1].shape[0]
            self.i += 1

        if self.i >= len(self.file_names) and self.loaded_len <= 0:
            raise StopIteration
        else:
            input_data = np.concatenate(self.loaded_in)
            output_data = np.concatenate(self.loaded_out)
            self.loaded_len -= min(self.batch_size, input_data.shape[0])
            self.loaded_out = [output_data[self.batch_size:]]
            self.loaded_in = [input_data[self.batch_size:]]

            return torch.from_numpy(input_data[:self.batch_size]), \
                    torch.from_numpy(output_data[:self.batch_size])


input_path = path_config['sensor_signal_path']
output_path = path_config['batched_pic_path']
batchsize = config['train']['batch_size']
test_dataloader = DataLoader(jn(input_path, 'test'), output_path)
train_dataloader = DataLoader(jn(input_path, 'train'),
                              output_path,
                              batch_size=batchsize)

# %% [markdown]
# # defining and fitting torch net

# %%
print('input batch shape: ',
      next(iter(train_dataloader))[0].shape, '\noutput batch shape: ',
      next(iter(train_dataloader))[1].shape)

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
# summary(model, next(iter(train_dataloader))[0].shape, device=device)
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
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            result.append(outputs)

    return torch.cat(result).numpy()


def iter_train(train_loader, test_loader, model, epochs, optimizer, criterion):
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
if not os.path.exists(path_config['reports_path']):
    os.makedirs(path_config['reports_path'])
df.to_csv(jn(path_config['reports_path'], 'learning_curve.csv'),
          index=False)
res = {'train': {'loss': train_loss[-1]}, 'test': {'loss': min(test_loss)}}
with open(jn(path_config['reports_path'], "summary.json"), "w") as f:
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
os.makedirs(path_config['model_path'])
torch.save(model, jn(path_config['model_path'], model_name + '.pt'))

# %% [markdown]
# # evaluate model on don't seen data

    # # %%
    # model = torch.load(jn(path_config['model_path'], model_name + '.pt'))
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
