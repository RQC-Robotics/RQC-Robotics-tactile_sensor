import torch
import torch.nn as nn
import torch.nn.functional as F


# deprecate
class FullyConnectedImprover(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(FullyConnectedImprover, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_shape[-1], 600), nn.ReLU(), nn.Linear(600, 1000),
            nn.ReLU(), nn.Linear(1000, 1000), nn.ReLU(), nn.Linear(1000, 600),
            nn.ReLU(), nn.Linear(600, output_shape[-1] * output_shape[-2]),
            nn.Unflatten(1, output_shape[-2:]))

    def forward(self, x):
        return self.sequential(x)


class TestModel(nn.Module):
    """Some Information about TestModel"""

    def __init__(self, pressure_shape, signal_shape):
        super(TestModel, self).__init__()
        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape

        self.sequential = nn.Sequential(
            nn.Linear(
                pressure_shape[-1] * pressure_shape[-2] +
                2 * signal_shape[-1] * signal_shape[-2], 600), nn.ReLU(),
            nn.Linear(600, 1000), nn.ReLU(), nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 600), nn.ReLU(),
            nn.Linear(600, pressure_shape[-1] * pressure_shape[-2]))

    def forward(self, previous_pressure, previous_signal, current_signal):
        y = torch.concat([
            torch.flatten(previous_pressure, 1),
            torch.flatten(previous_signal, 1),
            torch.flatten(current_signal, 1)
        ],
                         dim=-1)
        y = self.sequential(y)
        y = y.view(-1, *self.pressure_shape)
        return y


if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter

    model = TestModel((64, 64), (4, 64))
    writer = SummaryWriter('logdir')
    writer.add_graph(
        model,
        (torch.rand(1, 64, 64), torch.rand(1, 4, 64), torch.rand(1, 4, 64)))
    writer.close()
