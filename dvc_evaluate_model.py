# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jn
import yaml
import torch_sensor_lib as tsl

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd

# %%
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)
with open('pathes.yaml') as conf_file:
    path_config = yaml.safe_load(conf_file)

torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])
device = 'cpu'
model = torch.load(
    jn(path_config['model_path'], config['train']['model_name'] + '.pt'))
model.eval()
out_path = path_config['reports_path']


# %%
class MyDataSet(Dataset):

    def __init__(self, inputs: np.array, outputs: np.array):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


input_path = path_config['sensor_signal_path']
output_path = path_config['generated_pic_path']
file_name = os.listdir(input_path)[0]
inputs = np.load(jn(input_path, file_name))
outputs = np.load(jn(output_path, file_name))
test_size = config['train']['test_size']
if test_size == 'None':
    test_size = inputs.shape[0] // 20

# batchsize = config['train']['batch_size']
# train_dataloader = DataLoader(MyDataSet(inputs[:-test_size],
#                                         outputs[:-test_size]),
#                               batch_size=batchsize,
#                               shuffle=True)
test_dataset = MyDataSet(inputs[-test_size:],
                                       outputs[-test_size:])
test_dataloader = DataLoader(MyDataSet(inputs[-test_size:],
                                       outputs[-test_size:]),
                             batch_size=100)

# %%
report = open(jn(out_path, 'report.md'), 'w', encoding="utf-8")
print(f"# Report about training model **{config['train']['model_name']}**",
      file=report)
print(f"## Architecture summary\n```\n", file=report)
print(summary(
    model,
    next(iter(test_dataloader))[0].shape,
    device='cpu',
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    verbose=0),
      file=report)
print('\n```', file=report)

writer = SummaryWriter('logsdir')
# writer.add_graph(model, next(iter(test_dataloader))[0])
writer.close()
# %%
curve = pd.read_csv(jn(out_path, 'learning_curve.csv'))
curve.plot()
plt.title("Learning Curve")
plt.xlabel("epochs")
plt.ylabel("MSE loss")
plt.savefig(jn(out_path, 'l_curve.png'), dpi=100)
print("![learning curve](l_curve.png)", file=report)

# %% [markdown]
# ## Отображение разных примеров предсказания

# %%
# training functions


def picturewise_loss_and_predict(model, data_loader, criterion):
    losses = []
    result = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        losses.append(
            torch.mean(criterion(outputs, labels), dim=(-1, -2)).cpu().numpy())
        result.append(outputs.cpu().numpy())
    return np.concatenate(losses), np.concatenate(result)


# %%
losses, pred_pic = picturewise_loss_and_predict(
    model, test_dataloader, torch.nn.MSELoss(reduction='none'))
pred_pic.shape, losses.shape

# %%
import torch_sensor_lib as tsl


def create_examples_mesh(indecies, sample_titles):
    '''
    plots mesh of pictures, by indecies
    '''
    y_titles = ["true", "predict", "signal"]
    s = tsl.FiberSimulator(config, device='cpu')
    config['env']['phys']['noise'] = 0
    config['env']['phys']['relative_noise'] = 0

    data = []
    for ind in indecies:
        data.append([])
        signal, pic = test_dataset[ind]
        data[-1].append(pic)
        data[-1].append(pred_pic[ind])
        true_signal = signal[0]
        pred_signal = s._sum_fiber_losses(
            torch.from_numpy(pred_pic[ind:ind + 1]))[0].numpy()
        data[-1].append(
            np.concatenate(
                [true_signal.reshape(-1, 1),
                 pred_signal.reshape(-1, 1)],
                axis=1))

    visual_func = [lambda ax, pic: ax.imshow(pic)] * 2 + [
        lambda ax, pic:
        (ax.plot(pic, label=['received', 'predict']), ax.legend(loc="best"))
    ]
    tsl.visual_table_of_pictures(data, sample_titles, y_titles, visual_func)


# %%
best_ind = losses.argmin()
worst_ind = losses.argmax()
rand_ind1 = np.random.randint(len(losses))
rand_ind2 = np.random.randint(len(losses))
indexes = [best_ind, rand_ind1, rand_ind2, worst_ind]
sample_titles = ["best", "random", "random", "worst"]
y_titles = ["true", "predict", "signal"]

create_examples_mesh(indexes, sample_titles)

plt.savefig(jn(out_path, 'predict_examples.jpg'), dpi=50)

# more random predict examples
n = 10
indexes = np.random.randint(len(losses), size=n)
sample_titles = [f"loss={losses[i]:.3f}" for i in indexes]
create_examples_mesh(indexes, sample_titles)
plt.savefig(jn(out_path, 'rand_examples.jpg'), dpi=100)

# %%
print("## Examples of predictions", file=report)
print("![examples](predict_examples.jpg)", file=report)

# %%
report.close()
