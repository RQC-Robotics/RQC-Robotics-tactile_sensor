from flatten_dict import unflatten
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedImprover(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(FullyConnectedImprover, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_shape[-1], 600), nn.ReLU(),
            nn.Linear(600, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 600), nn.ReLU(),
            nn.Linear(600, output_shape[-1]*output_shape[-2]),
            nn.Unflatten(1, output_shape[-2:]))


    def forward(self, x):
        return self.sequential(x)


if __name__ == "__main__":
    from torchinfo import summary

    model = FullyConnectedImprover((2*64*(64+4),), (64, 64))

    summary(
        model, (5, 2*64*(64+4)),
        device='cpu',
        col_names=["input_size", "output_size", "num_params", "kernel_size"])