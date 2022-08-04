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
        super(ParamSingle, self).__init__()

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                signal_shape[-1] * signal_shape[-2], hidden_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Linear(hidden_layers[-1], 32*32)]

        self.sequential = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, previous_pressure, previous_signal, current_signal):
        x = torch.concat([torch.flatten(current_signal, 1)], dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 32, 32)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x

class ParamLatentRNN(nn.Module):
    def __init__(self, pressure_shape, signal_shape, hidden_layers: list[list[int]], memory_size):
        super(ParamLatentRNN, self).__init__()

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        self.memory_size = memory_size
        hidden_layers1 = hidden_layers[0]
        layers_first = [nn.Linear(
                signal_shape[-1] * signal_shape[-2]+memory_size, hidden_layers1[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers1[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers1)-1)]
                        
        self.sequential1 = nn.Sequential(*layers_first)

        hidden_layers2 = hidden_layers[1]
        layers_second = [nn.Linear(
                hidden_layers1[-1], hidden_layers2[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers2[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers2)-1)] + \
                        [nn.Linear(hidden_layers2[-1], 32*32)]
        
        self.sequential2 = nn.Sequential(*layers_second)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')
        self.memory = None
        
        self.to_memory = nn.Sequential(nn.Linear(hidden_layers1[-1], memory_size),
                                       nn.Sigmoid())
        
    def reset_memory(self):
        self.memory = None

    
    def forward(self, current_signal):
        if self.memory is None:
            self.memory = torch.zeros((current_signal.shape[-3], self.memory_size))
        x = torch.concat([
            self.memory,
            torch.flatten(current_signal, -2)
        ],
                         dim=-1)
        
        checkpoint1 = self.sequential1(x)
        self.memory = self.to_memory(checkpoint1)
        x = self.sequential2(checkpoint1)

        x = x.view(-1, 1, 32, 32)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)
        return x
    

if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter
    from torchinfo import summary

    model = ParamLatentRNN((64, 64), (4, 64), [[500, 500], [500, 500, 500]], memory_size=300)
    input_shape = (1, 4, 64)
    print(hasattr(model, "reset_memory"))
    summary(model, input_shape, col_names=["input_size", "output_size", "num_params"], device='cpu')
    
    # writer = SummaryWriter('logdir')
    # writer.add_graph(
    #     model,
    #     (torch.rand(*input_shape)))
    # writer.close()
