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
            nn.Linear(1000, 1000), nn.ReLU(), nn.Linear(1000, 30 * 30))

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, previous_pressure, previous_signal, current_signal):
        x = torch.concat([
            torch.flatten(previous_pressure, 1),
            torch.flatten(previous_signal, 1),
            torch.flatten(current_signal, 1)
        ],
                         dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 30, 30)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)
        return x

class SingleTestModel(nn.Module):

    def __init__(self, pressure_shape, signal_shape):
        super(SingleTestModel, self).__init__()
        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape

        self.sequential = nn.Sequential(
            nn.Linear(signal_shape[-1] * signal_shape[-2], 600), nn.ReLU(),
            nn.Linear(600, 1000), nn.ReLU(), nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(), nn.Linear(1000, 30 * 30))

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, previous_pressure, previous_signal, current_signal):
        x = torch.concat([torch.flatten(current_signal, 1)], dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 30, 30)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)
        return x


class ParamRNN(nn.Module):
    """Creates fully-connected net with certain layers"""

    def __init__(self, pressure_shape, signal_shape, hidden_layers: list[int]):
        super(ParamRNN, self).__init__()

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                pressure_shape[-1] * pressure_shape[-2] +
                2 * signal_shape[-1] * signal_shape[-2], hidden_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Linear(hidden_layers[-1], 30*30)]

        self.sequential = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, previous_pressure, previous_signal, current_signal):
        x = torch.concat([
            torch.flatten(previous_pressure, 1),
            torch.flatten(previous_signal, 1),
            torch.flatten(current_signal, 1)
        ],
                         dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 30, 30)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x


class ParamSingle(nn.Module):

    def __init__(self, pressure_shape, signal_shape, hidden_layers: list[int]):
        super(ParamRNN, self).__init__()

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                signal_shape[-1] * signal_shape[-2], hidden_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Linear(hidden_layers[-1], 30*30)]

        self.sequential = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, previous_pressure, previous_signal, current_signal):
        x = torch.concat([torch.flatten(current_signal, 1)], dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 30, 30)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x


if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter

    model = ParamRNN((64, 64), (4, 64), [600, 1000, 1000, 1000])
    writer = SummaryWriter('logdir')
    writer.add_graph(
        model,
        (torch.rand(1, 64, 64), torch.rand(1, 4, 64), torch.rand(1, 4, 64)))
    writer.close()
