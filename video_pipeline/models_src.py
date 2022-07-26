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

    def __init__(self, pressure_shape, signal_shape, hidden_layers: list[int], frames_number, frames_interval):
        super(ParamSingle, self).__init__()
        
        self.frames_interval, self.frames_number = frames_interval, frames_number 

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                signal_shape[-1] * signal_shape[-2], hidden_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Linear(hidden_layers[-1], 30*30)]

        self.sequential = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self,  current_signal):
        x = torch.concat([torch.flatten(current_signal, 1)], dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 30, 30)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x

class ParamStackNet(nn.Module):
    def __init__(self, pressure_shape, signal_shape, single_layers: list[int], shared_layers: list[int], frames_number, frames_interval):
        super(ParamStackNet, self).__init__()
        
        self.frames_interval, self.frames_number = frames_interval, frames_number 

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                signal_shape[-1] * signal_shape[-2], single_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*single_layers[i:i+2]), nn.ReLU()) for i in range(len(single_layers)-1)]

        self.sequential_single = nn.Sequential(*layers)
        
        layers = [nn.Linear(
                single_layers[-1]*frames_number, shared_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*shared_layers[i:i+2]), nn.ReLU()) for i in range(len(shared_layers)-1)] + \
                        [nn.Linear(shared_layers[-1], 32*32)]
        self.sequential_shared = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, signals_stack):
        props = []
        signals_stack = torch.swapdims(signals_stack, 0, 1)
        for signal in signals_stack:
            props.append(self.sequential_single(torch.flatten(signal, 1)))
        x = torch.concat(props, dim=-1)
        x = self.sequential_shared(x)
        x = x.view(-1, 1, 32, 32)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x


if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter
    from torchinfo import summary
    model = ParamStackNet((64, 64), (4, 64), [300, 100], [500, 200, 200, 500], 5, 5)
    print(model)
    summary(model, (1, 5, 4, 64), col_names=["input_size", "output_size", "num_params"], device='cpu')
    # writer = SummaryWriter('logdir')
    # writer.add_graph(
    #     model,
    #     (torch.rand(1, 5, 4, 64, device='cpu')))
    # writer.close()
