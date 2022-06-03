import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchSensorNN(nn.Module):

    def __init__(self, input_shape, output_shape):
        input_size = input_shape[-1] * input_shape[-2]
        super(TorchSensorNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(input_size, 600), nn.ReLU(),
            nn.Linear(600, 600), nn.ReLU(), nn.Linear(600, 600), nn.ReLU(),
            nn.Linear(600, output_shape[-1] * output_shape[-2]), nn.ReLU(),
            nn.Unflatten(1, output_shape[-2:]))

    def forward(self, x):
        return self.sequential(x)
